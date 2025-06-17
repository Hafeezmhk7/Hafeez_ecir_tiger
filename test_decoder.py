import argparse
import os
import gin
import torch
import wandb
import time
from accelerate import Accelerator
from data.processed import ItemData, RecDataset, SeqData
from data.utils import batch_to, cycle, next_batch, describe_dataloader
from evaluate.metrics import TopKAccumulator, FairnessAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics, parse_config, display_args, display_metrics, display_model_summary, set_seed
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from tqdm import tqdm
from rich.logging import RichHandler
import logging


# logging
os.environ["WANDB__SERVICE_WAIT"] = "300"

# create logger
logger = logging.getLogger("recsys_logger")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = RichHandler(show_path=False)
    logger.addHandler(handler)
    logger.propagate = False
    

def create_item_brand_mapping(tokenizer, item_dataset):
    item_brand_mapping = {}
    
    try:
        if hasattr(tokenizer, 'cached_ids') and tokenizer.cached_ids is not None:
            semantic_ids = tokenizer.cached_ids[:, :-1]
            
            for idx, semantic_id in enumerate(semantic_ids):
                if idx < len(item_dataset.item_brand_id):
                    brand_id = item_dataset.item_brand_id[idx]
                    semantic_id_tuple = tuple(semantic_id.tolist())
                    item_brand_mapping[semantic_id_tuple] = brand_id
            
            logger.info(f"Created brand mapping for {len(item_brand_mapping)} items")
            
    except Exception as e:
        logger.warning(f"Could not create item-brand mapping: {e}")
    
    return item_brand_mapping

def clamp_ids(tokenized_data, valid_max):
    valid_sem_id_min = tokenized_data.sem_ids.min().item()
    valid_sem_id_fut_min = tokenized_data.sem_ids_fut.min().item()
    tokenized_data = tokenized_data._replace(
        sem_ids=torch.clamp(tokenized_data.sem_ids, min=valid_sem_id_min, max=valid_max),
        sem_ids_fut=torch.clamp(tokenized_data.sem_ids_fut, min=valid_sem_id_fut_min, max=valid_max)
    )
    return tokenized_data


def evaluate(model, test_dataloader, device, tokenizer, 
             metrics_accumulator, fairness_accumulator,
             item_dataset, 
             use_image_features):
    # set model to evaluation mode
    model.eval()
    total_loss = 0
    debug_metrics = []
    num_batches = 0
    pbar = tqdm(test_dataloader, desc=f"Eval")
    
    # setup fairness accumulator
    item_brand_mapping = None
    if fairness_accumulator is not None:
        item_brand_mapping = create_item_brand_mapping(tokenizer, item_dataset)
        
        if item_brand_mapping:
            try:
                if hasattr(item_dataset, 'data') and 'brand_mapping' in item_dataset.data:
                    brand_mapping = item_dataset.data['brand_mapping']
                else:
                    brand_ids = set(item_dataset.item_brand_id[item_dataset.item_brand_id >= 0])
                    brand_mapping = {brand_id: f"Brand_{brand_id}" for brand_id in brand_ids}
                
                fairness_accumulator.set_brand_mappings(item_brand_mapping, brand_mapping)
                fairness_accumulator.set_auto_groups(dataset_folder=None, dataset_split=None)
                logger.info("Fairness accumulator configured with brand mappings")
            except Exception as e:
                logger.warning(f"Could not configure fairness accumulator: {e}")
                
    for batch in pbar:
        data = batch_to(batch, device)
        tokenized_data = tokenizer(data)
        # clamp semids
        valid_max = model.num_embeddings - 1
        tokenized_data = clamp_ids(tokenized_data, valid_max)
        model.enable_generation = False
        # debug metrics
        with torch.no_grad():
            model_output_eval = model(tokenized_data)
            loss = model_output_eval.loss.detach().cpu().item()
            total_loss += loss
            num_batches += 1
            eval_debug_metrics = compute_debug_metrics(
                tokenized_data, model_output_eval, "eval"
            )
            debug_metrics.append(eval_debug_metrics)
        # eval metrics
        model.enable_generation = True
        generated = model.generate_next_sem_id(
            tokenized_data, top_k=True, temperature=1
        )
        actual, top_k = tokenized_data.sem_ids_fut, generated.sem_ids
        # calculate IR measures
        metrics_accumulator.accumulate(
            actual=actual, top_k=top_k, tokenizer=tokenizer
        )
        # calculte fairness metrics
        fairness_accumulator.accumulate(
                    actual=actual, 
                    top_k=top_k, 
                    user_ids=tokenized_data.user_ids,
                    performance_scores=None,
                    item_brand_mapping=item_brand_mapping)
        
    eval_metrics = metrics_accumulator.reduce()
    eval_metrics = {f"metrics/{k}": v for k, v in eval_metrics.items()}
    # reset the metrics accumulator
    metrics_accumulator.reset()
    
    fairness_metrics = fairness_accumulator.reduce()
    fairness_metrics_prefixed = {f"fairness/{k}": v for k, v in fairness_metrics.items()}
    eval_metrics.update(fairness_metrics_prefixed)
    fairness_accumulator.reset()
    
    # average debug metrics
    averaged_debug_metrics = {}
    for key in debug_metrics[0].keys():
        averaged_debug_metrics[key] = sum(d[key] for d in debug_metrics) / num_batches

    averaged_debug_metrics["eval/loss"] = total_loss / num_batches
    eval_metrics.update(averaged_debug_metrics)
            
    return eval_metrics


@gin.configurable
def test(
    batch_size=64,
    dataset_folder="dataset/ml-1m",
    log_dir="out/",
    dataset=RecDataset.ML_1M,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    decoder_embed_dim=64,
    dropout_p=0.1,
    attn_heads=8,
    attn_embed_dim=64,
    attn_layers=4,
    dataset_split="beauty",
    train_data_subsample=True,
    model_jagged_mode=True,
    category=None,
    use_image_features=False,
    feature_combination_mode="sum",
    run_prefix="",
    debug=False,
):

    # create logdir if not exists
    uid = str(int(time.time()))
    logger.info(
        f"Session Started with UID '{uid}' | Dataset '{dataset_folder}' | Split '{dataset_split}'")
    # log_dir = os.path.join(os.path.expanduser("~"), log_dir, dataset_split, uid)
    # os.makedirs(log_dir, exist_ok=True)

    if dataset != RecDataset.AMAZON:
        raise Exception(f"Dataset currently not supported: {dataset}.")

    # setup accelerator and device
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device
    display_args(locals())

    if pretrained_rqvae_path is None or pretrained_decoder_path is None:
        logger.error("No pretrained rqvae or decoder path provided. Please provide valid paths to continue training.")
        return

    # extract rq-vae uid
    rqvae_uid = pretrained_rqvae_path.split("/")[-2]
    
    # logging
    if wandb_logging and accelerator.is_main_process:
        # get local scope parameters for logging
        params = locals()
        # wandb.login()
        run_name = f"decoder-{dataset.name.lower()}-{dataset_split}" + \
            "/" + rqvae_uid + "/" + uid
        if run_prefix:
            run_name = f"{run_prefix}-{run_name}"
        run = wandb.init(entity="RecSys-UvA",
                         name=run_name,
                         project="gen-ir-decoder-testing",
                         config=params)
                         
    # load items dataset
    item_dataset = (
        ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=force_dataset_process,
            split=dataset_split,
            use_image_features=use_image_features,
            feature_combination_mode=feature_combination_mode,
            device=device,
        )
        if category is None
        else ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=force_dataset_process,
            split=dataset_split,
            category=category,
            use_image_features=use_image_features,
            feature_combination_mode=feature_combination_mode,
            device=device,
        )
    )
    # load train dataset
    train_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        subsample=train_data_subsample,
        split=dataset_split,
        data_split="train",
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = cycle(train_dataloader)

    # load test dataset
    test_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        subsample=False,
        split=dataset_split,
        data_split="test",
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    describe_dataloader(test_dataloader, title="Test DataLoader Summary")
    train_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, test_dataloader
    )

    # load rq-vae tokenizer
    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
    )
    tokenizer = accelerator.prepare(tokenizer)
    tokenizer.precompute_corpus_ids(item_dataset)

    # load model
    model = EncoderDecoderRetrievalModel(
        embedding_dim=decoder_embed_dim,
        attn_dim=attn_embed_dim,
        dropout=dropout_p,
        num_heads=attn_heads,
        n_layers=attn_layers,
        num_embeddings=vae_codebook_size,
        inference_verifier_fn=lambda x: tokenizer.exists_prefix(x),
        sem_id_dim=tokenizer.sem_ids_dim,
        max_pos=train_dataset.max_seq_len * tokenizer.sem_ids_dim,
        jagged_mode=model_jagged_mode,
    )
    display_model_summary(model, device)

    if pretrained_decoder_path is not None:
        logger.info(
            f"Loading pretrained Decoder from {pretrained_decoder_path}.")
        checkpoint = torch.load(
            pretrained_decoder_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
    else:
        logger.error("No pretrained decoder path provided. Please provide a valid path to continue testing.")
        return

    model = accelerator.prepare(model)

    metrics_accumulator = TopKAccumulator(ks=[1, 5, 10])
    fairness_accumulator = FairnessAccumulator(ks=[1, 5, 10])
    
    # starting the testing
    logger.info(f"Testing Started! - Debugging: {debug}")
    eval_log = evaluate(model, test_dataloader, device, tokenizer, 
                        metrics_accumulator, fairness_accumulator, 
                        item_dataset,
                        use_image_features)
    
    # print eval metrics
    display_metrics(eval_log, title="Testing Metrics")
    
    # log metrics
    if wandb_logging:
        wandb.log(eval_log)
                

    if wandb_logging:
        wandb.finish()


if __name__ == "__main__":
    set_seed(42)
    parse_config()
    test()
