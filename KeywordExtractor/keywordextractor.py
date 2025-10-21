from abc import abstractmethod
from keybert import KeyBERT

from util.preprocess_constant import ObjectType, Modality, LLMName

sample_query = '''Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal).
         A supervised learning algorithm analyzes the training data and produces an inferred function,
         which can be used for mapping new examples. An optimal scenario will allow for the
         algorithm to correctly determine the class labels for unseen instances. This requires
         the learning algorithm to generalize from the training data to unseen situations in a
         'reasonable' way (see inductive bias).'''




class KeywordExtractorBuilder:

    @staticmethod
    def build(name, **kwargs):
        if name == 'KeyBERT':
            return KeyBertKeywordExtractor(**kwargs)
        elif name in LLMName.all_llms:
            return LLMKeywordExtractor(**kwargs)
        elif name == 'Bare':
            return BareKeywordExtractor()
        elif name == 'Non':
            return NonKeywordExtractor()
        else:
            raise NotImplementedError


class BaseKeywordExtractor(object):

    @abstractmethod
    def retrieve(self, query, doc, modalities):
        pass

    @abstractmethod
    def extract_keywords(self, query, modalities):
        pass

    @abstractmethod
    def warm_up(self):
        pass
class NonKeywordExtractor(BaseKeywordExtractor): # 不做retrieve，直接返回modality所有的内容
    def __init__(self):
        self.name = 'Non'

    def retrieve(self, query, doc, modalities):
        pass

    def extract_keywords(self, query, modalities):
        modality2keywords = {}
        for modality in Modality.RetrieveModalities:
            modality2keywords[modality] = None
        modality2keywords['DET'] = None
        modality2keywords['TYPE'] = ObjectType.all_types
        return modality2keywords

class BareKeywordExtractor(BaseKeywordExtractor):  # 直接用整个query进行retrieve
    def __init__(self, **kwargs):
        self.name = 'Bare'
    def retrieve(self, query, doc, modalities):
        pass

    def extract_keywords(self, query, modalities):
        modality2keywords = {}
        for modality in Modality.RetrieveModalities:
            modality2keywords[modality] = query
        modality2keywords['DET'] = [query]
        modality2keywords['TYPE'] = ObjectType.all_types
        return modality2keywords

    def warm_up(self):
        pass


class SimpleKeywordKeywordExtractor(BaseKeywordExtractor):  # 考虑一些rule-based keyword extractor进行retrieve
    def retrieve(self, query, doc):
        pass

    def extract_keywords(self, query, doc):
        pass


class KeyBertKeywordExtractor(BaseKeywordExtractor):
    def __init__(self, **kwargs):
        self.max_keyword = 5
        self.keyphrase_ngram_range = (1, 3)
        if kwargs.get('max_keyword'):
            self.max_keyword = kwargs['max_keyword']
        if kwargs.get('keyphrase_ngram_range'):
            self.keyphrase_ngram_range = kwargs['keyphrase_ngram_range']
        self.name = 'KeyBERT'
        self.model = KeyBERT()



    def retrieve(self, query, doc, modalities):

        pass


    def extract_keywords(self, query, modalities):
        keywords_with_freq = self.model.extract_keywords(query, keyphrase_ngram_range=self.keyphrase_ngram_range)
        keywords_with_freq = keywords_with_freq[:min(self.max_keyword, len(keywords_with_freq))]
        modality2keywords = {}
        for modality in Modality.RetrieveModalities:
            modality2keywords[modality] = [item[0] for item in keywords_with_freq]
        modality2keywords['DET'] = modality2keywords[Modality.Object]
        modality2keywords['TYPE'] = ObjectType.all_types
        return modality2keywords

    def warm_up(self):
        print(f'Retriever {self.name} start to prewarm...')
        keywords = self.model.extract_keywords(sample_query)
        print(f'{sample_query} keywords: {keywords}')
        print(f'Retriever {self.name} prewarm finished.')

class LLMKeywordExtractor(BaseKeywordExtractor):  # 使用LLM进行retrieve
    def __init__(self, **kwargs):
        self.name = kwargs['llm_name']
        self.prompt_factory = kwargs['prompt_factory']
        self.llm_gen_func = kwargs['llm_gen_func']

    def retrieve(self, query, doc, modalities):

        pass

    def warm_up(self):
        self.llm_gen_func('Who are you?', video=None)

    def extract_keywords(self, query, modalities):
        prompt = self.prompt_factory.retrieve_prompt(query)
        return self.llm_gen_func(prompt, video=None)


class VLMTuneKeywordExtractor(BaseKeywordExtractor):
    def __init__(self, model):
        self.model = model
        pass

    def retrieve(self, query, doc):
        pass

    def extract_keywords(self, query, doc):
        pass
