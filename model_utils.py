import os
import logging
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models import Model
from modelscope.utils.constant import Tasks

logger = logging.getLogger(__name__)

def download_model(
    model_id: str,
    model_revision: Optional[str] = None,
    cache_dir: str = "pretrained_models",
    source: str = "huggingface"
) -> Tuple[str, bool]:
    """
    从Hugging Face或ModelScope下载模型。
    
    Args:
        model_id: 模型ID，例如'langboat/mengzi-t5-base'或'damo/nlp_t5_summary-translation_chinese'
        model_revision: 模型版本，默认为None
        cache_dir: 模型缓存目录，默认为'pretrained_models'
        source: 模型来源，可选'huggingface'或'modelscope'
    
    Returns:
        Tuple[str, bool]: (模型本地路径, 是否来自modelscope)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # 构建模型本地保存路径
    model_name = model_id.split('/')[-1]
    local_dir = os.path.join(cache_dir, f"{source}-{model_name}")
    
    if os.path.exists(local_dir):
        logger.info(f"模型已存在于本地: {local_dir}")
        return local_dir, source == "modelscope"
    
    try:
        if source == "huggingface":
            logger.info(f"从Hugging Face下载模型: {model_id}")
            # 下载模型和tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                revision=model_revision
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                revision=model_revision
            )
            
            # 保存到指定目录
            os.makedirs(local_dir, exist_ok=True)
            tokenizer.save_pretrained(local_dir)
            model.save_pretrained(local_dir)
            
        elif source == "modelscope":
            logger.info(f"从ModelScope下载模型: {model_id}")
            local_dir = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                revision=model_revision
            )
            
        else:
            raise ValueError(f"不支持的模型来源: {source}")
        
        logger.info(f"模型已下载到: {local_dir}")
        return local_dir, source == "modelscope"
    
    except Exception as e:
        logger.error(f"下载模型时出错: {str(e)}")
        if source == "huggingface":
            logger.info("尝试从ModelScope下载...")
            return download_model(model_id, model_revision, cache_dir, "modelscope")
        raise

def load_model_and_tokenizer(
    model_path: str,
    is_modelscope: bool = False
) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """
    加载模型和tokenizer。
    
    Args:
        model_path: 模型路径
        is_modelscope: 是否是ModelScope模型
    
    Returns:
        Tuple[Model, AutoTokenizer]: (模型, tokenizer)
    """
    try:
        if is_modelscope:
            model = Model.from_pretrained(model_path, task=Tasks.text_generation)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        raise
