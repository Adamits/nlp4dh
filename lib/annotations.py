from allennlp.common.file_utils import cached_path
from allennlp.service.predictors import SemanticRoleLabelerPredictor
from allennlp.models.archival import load_archive

import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
from datetime import datetime
import argparse

DEV_MODE = False
if DEV_MODE:
    NLP4DH_PATH = "/Users/ajwieme/nlp4dh"
else:
    NLP4DH_PATH = "/nlp4dh"

# Tab delimited file of hardcoded mappings of
# propbank role to SRL tag name
MAPPING_FILE = NLP4DH_PATH + "/lib/static/srl_mappings"

def get_date_from_string(s):
    """
    Turns a string in the format of a numerical year,
    e.g. 1991, into a datetime object
    """
    try:
        # This will error if it is not a datetime of that format
        datetime.strptime(s, "%Y")
        return s
    except:
        return None

def doc2json(doc):
    """
    Given a spaCy doc, return a tuple of
    (sentencized spaCy tokens, json in the form:
    [{"sentence": "..."}, {"sentence": "..."}])
    """
    sentences = [s for s in doc.sents]
    json_sents = [{"sentence": sent.string.strip()}\
                       for sent in doc.sents]

    return (sentences, json_sents)

def get_srl_model():
    """
    Makes sure file exists and then returns the path

    HARDCODED FOR NOW
    """
    return cached_path(NLP4DH_PATH + '/models/srl-model-2018.02.27.tar.gz')

def get_srl_mapping():
    """
    Returns a mapping dict based on MAPPING_FILE
    """
    lines = [l.strip() for l in open(MAPPING_FILE)]
    mappings = {l.split('\t')[0]: l.split('\t')[1] for l in lines}

    return mappings

def anlp2srl(sentence, v_tags, mapping):
    """
    Given the anlp sequences of SRL propbank tags,
    Map to the appropriate SRL tag, and use BIO tags
    to structure each srl tag as a {tag: text} combination

    Note words should be a spaCy span of tokens

    Return: [{tag: [words]}, {tag: [words]}]
    """
    def get_tagname(tag):
        if '-' in tag:
            return tag.split('-')[1]
        else:
            return tag

    def is_phrase_chunk(last):
        return last.startswith('B') or last.startswith('I')

    def is_end_of_chunk(current, next):
        return not next.startswith('I') or get_tagname(current) != get_tagname(next)

    verb, tags = v_tags
    last_t = ''
    current_chunk = ()
    chunks = []
    if len(sentence) != len(tags):
        raise Exception("Something is up with: %s, %s of lengths: %i and %i!"\
             % (sentence.text, ','.join(tags), len(words), len(tags)))

    for i, t in enumerate(tags):
        # Check if this should be appended to the current phrase
        if is_phrase_chunk(last_t) and get_tagname(t) == get_tagname(last_t):
            current_chunk[1].append(i)
        else:
            current_chunk = (get_tagname(t), [i])
        # Check if this is the end of a single tagged word or phrase
        if i < len(tags)-1 and is_end_of_chunk(t, tags[i+1]):
            span = [current_chunk[1][0], current_chunk[1][-1]]
            # (tag, content (+1 to include last slot in the span), span)
            c = (current_chunk[0], sentence[span[0]:span[1] + 1], span)
            chunks.append(c)

        last_t = t

    return [{mapping[t]: {"parent": verb, "content": w.text, "span": s}}\
            for t, w, s in chunks if t in mapping.keys()]

def get_spans(args):
    """
    This method should loop through the annotations of a sentence, and combine
    them into 'textSpans': chunks of text that have some lingusitic
    annotation. Each textSpan will have a key for every annotation type.
    """
    spans = {}
    srl_list = args.get("srl")
    # [tag: {parent: ...}, content: ..., span: [x, y]]
    if srl_list is not None:
        for srl_dict in srl_list:
            for tag, data in srl_dict.items():
                srl = {tag: {"parent": data.get("parent")}}
                span = spans.get(tuple(data.get("span")))
                if span is not None:
                    span.update({"srl": srl})
                else:
                    spans[tuple(data.get("span"))] = {"srl": srl, "content": data.get("content")}

    flattened_spans = []
    for span, span_data in spans.items():
        span_data.update({"span": list(span)})
        flattened_spans.append(span_data)

    return flattened_spans


def annotations2json(fn, date, sentences, srl_sentences):
    """
    Returns a JSON of the AllenNLP output JSON,
    which can be indexed in the elasticsearch backend

    Sentences should be a spacy span of tokens
    """
    name = fn.split('/')[-1]
    sentence_jsons = {
        "name": name,
        # Right now, were only dealing with years
        "year": date,
        "sentences": []
    }
    mapping = get_srl_mapping()

    for i, sentence in enumerate(sentences):
        # Get a list of tag lists
        srls = [(v['verb'], v['tags']) for v in srl_sentences[i]['verbs']]
        # Update it to the desired output.
        srl = [s for srl in srls for s in anlp2srl(sentence, srl, mapping)]
        sentence_json = {
            "content": sentence.string.strip(),
            "textSpans": get_spans({"srl": srl})
        }

        sentence_jsons["sentences"].append(sentence_json)

    return sentence_jsons

def make_annotation_json(fn):
    lines = [l.strip() for l in open(fn)]
    # Look for a year in the first line
    date = get_date_from_string(lines[0])
    if date:
        # If the first line is a year,
        # then ignore that line and get
        # the rest of the doc as a string
        text = ' '.join(lines[1:])
    else:
        text = ' '.join(lines)

    # Get spacy doc
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(text)

    # Get sentencized text, and json format for AllenNLP
    sentences, json_sentences = doc2json(doc)
    archive = load_archive(get_srl_model())
    predictor = SemanticRoleLabelerPredictor.\
                from_archive(archive, "semantic-role-labeling")
    srl_sents = predictor.predict_batch_json(json_sentences)

    #print(annotations2json(fn, sentences, srl_sents))
    return annotations2json(fn, date, sentences, srl_sents)

def bulk_make_annotation_json(fns=[]):
    archive = load_archive(get_srl_model())
    predictor = SemanticRoleLabelerPredictor.\
                from_archive(archive, "semantic-role-labeling")
    jsons = []
    for fn in fns:
        lines = [l.strip() for l in open(fn)]
        # Look for a year in the first line
        date = get_date_from_string(lines[0])
        if date:
            # If the first line is a year,
            # then ignore that line and get
            # the rest of the doc as a string
            text = ' '.join(lines[1:])
        else:
            text = ' '.join(lines)

        # Get spacy doc
        nlp = English()
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        doc = nlp(text)

        # Get sentencized text, and json format for AllenNLP
        sentences, json_sentences = doc2json(doc)
        srl_sents = predictor.predict_batch_json(json_sentences)

        #print(annotations2json(fn, sentences, srl_sents))
        jsons.append(annotations2json(fn, date, sentences, srl_sents))

    return jsons

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Get the json with all of the annotations")
    parser.add_argument('--fn', help="filename to annotate", required=True)
