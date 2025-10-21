
class KnobType:
    IntKnob = 'IntKnob',
    FloatKnob = 'FloatKnob',
    CategoricalKnob = 'CategoricalKnob',

class Modality:
    VisionToken = 'visiontoken'
    Caption = 'caption'
    Frame = 'frame'
    Object = 'object'
    ASR = 'asr'
    OCR = 'ocr'
    Clip = 'clip'
    # visiontoken_config, caption_config, object_config, ocr_config, asr_config
    MainModalities = [VisionToken, Caption, Object, OCR, ASR]
    LoadModalities = [VisionToken, Object, OCR, ASR] # inference时需要额外时间开销（如load from disk）的modality
    RetrieveModalities = [Object, OCR, ASR] # 需要LLM进行retrieve的modality
    PreprocessModalities = [VisionToken, Caption,Object, OCR, ASR] # 需要LLM进行retrieve的modality
    BasicModalities = [VisionToken, Caption]
    NonLLMModalities = [Clip, Frame]

class KeyWordExtractorNames:
    LlavaVideoQwen7b = 'LlavaVideoQwen7b'
    VideoLlamaMistral7b = 'VideoLlamaMistral7b'
    LLMExtractor = 'LLM'
    KeyBertExtractor = 'KeyBERT'
    BareExtractor = 'Bare'
    NonExtractor = 'Non'
    all_extractor_names = [LLMExtractor, KeyBertExtractor, BareExtractor,NonExtractor]

class CaptionModelName:
    llava = 'llava'
    lavila = 'lavila'
    all_models = [llava, lavila]

class ASRModelName:
    whisper_small = 'whisper-small'
    whisper_medium = 'whisper-medium'
    whisper_large = 'whisper-large'
    all_models = [whisper_small, whisper_medium, whisper_large]

class QuestionType:
    OE = 'OE'  # open-ended question
    MC = 'MC'  # multiple-choice question

class ObjectType:
    Location = 'location'
    Relation = 'relation'
    Number = 'number'
    all_types = [Location, Relation, Number]

class ObjModelName:
    RTDETR = 'RTDETR'
    APE_L_D = 'APELD'
    ape_series_model_names = [APE_L_D]
    ultralytics_series_model_names = [RTDETR]
    all_models = [APE_L_D, RTDETR]

class PrepConfigSetType:
    ObjectOnly = 'ObjectOnly'
    Combination = 'Combination'
    CaptionOnly = 'CaptionOnly'
    VisiontokenOnly = 'VisiontokenOnly'
    All = 'All'
    all_three_types = [Combination, CaptionOnly, VisiontokenOnly]
    Combination_types = [Combination]
    CaptionOnly_types = [CaptionOnly]
    VisiontokenOnly_types = [VisiontokenOnly]


class VQADataset:
    CinePipe = 'CinePipe'
    MSVD_QA = 'MSVD-QA'
    ALLVB = 'ALLVB'
    AGQA = 'AGQA'
    NextQA = 'Next-QA'
    SampleVideo = 'sample_video'
    MSRVTT_QA = 'MSRVTT-QA'
    GroundQA = 'GroundQA'

class LLMName:
    LlavaVideoQwen7b = 'LlavaVideoQwen7b'
    VideoLlamaMistral7b = 'VideoLlamaMistral7b'
    all_llms = [LlavaVideoQwen7b, VideoLlamaMistral7b]

modality2priority = {
    Modality.VisionToken: 0,
    Modality.Caption: 1,
    Modality.Object: 2,
    Modality.ASR: 3,
    Modality.OCR: 4,
}


def sort_modalities(modalities: list) -> list:
    return sorted(modalities, key=lambda mod: modality2priority[mod])


def modality2abbr(modalities):
    modality_abbr_dict = {
        Modality.VisionToken: 'VT',
        Modality.Caption: 'C',
        Modality.Object: 'OB',
        Modality.ASR: 'A',
        Modality.OCR: 'OC',
    }
    if isinstance(modalities, list):
        return [modality_abbr_dict[modality] for modality in modalities]
    elif isinstance(modalities, str):
        return modality_abbr_dict[modalities]
    else:
        raise TypeError
