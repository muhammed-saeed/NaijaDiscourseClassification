import os
from argparse import ArgumentParser
import re
import sys
import json
import trankit
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic.main import BaseModel
# from simpletransformers.t5 import T5Model, T5Args
from discopy_data.data.loaders.raw import load_textss as load_texts_fast
from discopy.parsers.pipeline import ParserPipeline
from discopy_data.nn.bert import get_sentence_embedder
import pandas as pd
import supar
from discopy_data.data.update import get_constituent_parse, get_dependency_parse
   
# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)

# model_args = T5Args()
# model_args.max_length = 256
# model_args.n_gpu = 4
# model_args.length_penalty = 1
# model_args.num_beams = 10
# model_args.use_multiprocessed_decoding = False
# model_output_dir = "/local/musaeed/BESTT5TranslationModel"
# model = T5Model("t5", model_output_dir, args=model_args, use_cuda=True)

arg_parser = ArgumentParser()
arg_parser.add_argument("--model-path", default="/local/musaeed/discopy_models", type=str, help="path to trained discourse parser")
arg_parser.add_argument("--bert-model", default='bert-base-cased', type=str, help="bert model name")
args = arg_parser.parse_args()

model_path = "/home/CE/musaeed/bert_model/"


df = pd.read_csv("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/csv/processed/filteredTESTDEV.csv")
# df = pd.read_csv("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/dev/data/pcmTreeBankDevTestData.csv")
# english_translatedData = df['EnglishTranslationPCM'].tolist()
df = pd.read_csv("/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/dev/data/csv/devTestDatasetwithTranslationsOfPCM.csv")
english_real_annotation = df['EnglishTranslationPCMWithoutDEVTest'].tolist()

parser: ParserPipeline = None
get_sentence_embeddings = None

class ParserRequest(BaseModel):
    details: str
    title: str

def load_parser(model_path):
    parser = ParserPipeline.from_config(model_path)
    parser.load(model_path)
    return parser

constituent_parser='crf-con-en'
dependency_parser='biaffine-dep-en'
constituents=True
dependencies=True
cparserA = supar.Parser.load(constituent_parser) if constituents else None
dparserA = supar.Parser.load(dependency_parser) if dependencies else None


def load_parsertrankit(use_gpu=False):
    import trankit
    tmp_stdout = sys.stdout
    sys.stdout = sys.stderr
    parser = trankit.Pipeline('english', cache_dir=os.path.expanduser('~/.trankit/'), gpu=use_gpu)
    parser.tokenize("Init")
    sys.stdout = tmp_stdout
    print("we have completed the load_paraser step", file=sys.stderr)
    return parser
parserA = load_parsertrankit()

def tokenize(text, fast=False, tokenize_only=False):
    from discopy_data.data.loaders.raw import load_textss as load_texts_fast
    from discopy_data.data.loaders.raw import load_textsParserArgument
    output = []
    arr2text = ". ".join(text)
    document_loader = load_texts_fast if fast else load_textsParserArgument
    document_loader = load_textsParserArgument
    for doc in document_loader(re.split(r'\n\n\n+', text), parserA, tokenize_only=False):
        output.append(doc)
    return output

def add_parsers(src,cparser=cparserA,dparser=dparserA, constituent_parser='crf-con-en', dependency_parser='biaffine-dep-en', constituents=True, dependencies=True):
    # cparser = supar.Parser.load(constituent_parser) if constituents else None
    # dparser = supar.Parser.load(dependency_parser) if dependencies else None
    output = []
    for doc in src:
        for sent_i, sent in enumerate(doc.sentences):
            inputs = [(t.surface, t.upos) for t in sent.tokens]
            if cparser:
                parsetree = get_constituent_parse(cparser, inputs)
                doc.sentences[sent_i].parsetree = parsetree
            if dparser:
                dependencies = get_dependency_parse(dparser, inputs, sent.tokens)
                doc.sentences[sent_i].dependencies = dependencies
        output.append(doc)
    return output



def apply_parseren(r, parser):
    get_sentence_embeddings = get_sentence_embedder(args.bert_model)
    enTEXT = r.details
    text_array = enTEXT.split(".")
    text_array = list(filter(lambda x: x.strip(), text_array))
    translation = ". ".join(text_array) + "."
    doc = add_parsers(tokenize(str(translation)))[0]
    if len(doc.sentences) == 0:
        return {"translatedDetails": str("You are passing an empty string ;)")}

    for sent_i, sent in enumerate(doc.sentences):
        sent_words = sent.tokens
        embeddings = get_sentence_embeddings(sent_words)
        doc.sentences[sent_i].embeddings = embeddings
    doc = parser(doc)
    doc_json = doc.to_json()
    return {"English Annotated Sentences Parsed": doc_json}

if __name__ == '__main__':
    print("you are welcome here")
    
    parser = load_parser(args.model_path)
    tmp_stdout = sys.stdout
    sys.stdout = sys.stderr
    parserForLoadText = trankit.Pipeline('english', cache_dir=os.path.expanduser('~/.trankit/'), gpu=False)
    parserForLoadText.tokenize("Init")
    sys.stdout = tmp_stdout
    print("we have completed the load_parser step", file=sys.stderr)
    
    # Test sentence
    # sentences = ["the weather is nice. however Idon't have time", "however the weather is not  nice. I am not going outside"]
    output = []
    english_real_annotation = ["The girl i love doesnot allow me to tell her I love her. However i love her.", "I am going outside.Since the wehter is nice" ]
    for idx,sentence in enumerate(english_real_annotation):
        request = ParserRequest(details=sentence, title="")
        result = apply_parseren(request, parser)
        output.append(result)

        with open(f"/local/musaeed/NaijaDiscourseClassification/TreeBankAnnotated/parsedDataDiscopy/devTest/fake/{idx}.json", "w") as file:
            json.dump(output, file)
