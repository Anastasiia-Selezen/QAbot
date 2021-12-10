import os
from typing import List, Dict
import re
import lxml.etree
from io import StringIO

from haystack.document_store import ElasticsearchDocumentStore
from haystack.retriever.dense import DensePassageRetriever


def read_file(name: str) -> List[str]:
    with open(name) as f:
        lines = f.readlines()
        return lines


def extract_chapters(lines: List[str]) -> List[Dict[str, str]]:
    book_regexp = re.compile('\w* EPILOGUE|BOOK \w*:\d*')
    chapter_regexp = re.compile('Chapter \w*')
    documents = []
    book = None
    chapter = None
    text = ''
    for line in lines:
        if book_regexp.match(line):
            if book is not None and chapter is not None:
                documents.append({"text": text.strip(), "meta": {"name": f'{book} {chapter}'}})
            book = line.split(':')[0].strip()
            chapter = None
            text = ''
            continue
        if chapter_regexp.match(line):
            if book is not None and chapter is not None:
                documents.append({"text": text.strip(), "meta": {"name": f'{book} {chapter}'}})
            chapter = line.strip()
            text = ''
            continue
        if not line.isspace():
            line = " ".join(line.split()) + " "
            text += line
    documents.append({"text": text.strip(), "meta": {"name": f'{book} {chapter}'}})
    return documents


def extract_xml_chapters(lines: str) -> List[Dict[str, str]]:
    documents = []
    parser = lxml.etree.XMLParser(ns_clean=True, recover=True, encoding='utf-8')
    root = lxml.etree.parse(StringIO(lines), parser)
    for book in list(root.getroot())[1].xpath("./*[name()='section']"):
        book_name = ''.join(book[0].itertext()).strip()
        if book_name.isspace():
            break
        for part in book.xpath("./*[name()='section']"):
            part_name = ''.join(part[0].itertext()).strip()
            for chapter in part.xpath("./*[name()='section']"):
                chapter_name = ''.join(chapter[0].itertext()).strip()
                print(f'{book_name} {part_name} {chapter_name}')
                text = ''
                for para in chapter.xpath("./*[name()='p' or name()='poem']"):
                    para_text = ''.join(para.itertext())
                    text += ' '.join(para_text.split()) + ' '
                documents.append({"text": text.strip(), "meta": {"name": f'{book_name} {part_name} {chapter_name}'}})
    return documents


def preprocess(documents: List[Dict[str, str]]):
    elastic_host = 'localhost'
    elastic_embedding_dim = 768
    store = ElasticsearchDocumentStore(
        host=elastic_host,
        username="",
        password="",
        index="document",
        embedding_dim=elastic_embedding_dim,
        embedding_field="embedding")

    retriever = DensePassageRetriever.load(load_dir='../QA_models/dpr',
                                           document_store=store,
                                           infer_tokenizer_classes=True,
                                           use_gpu=False)

    store.delete_documents()
    store.write_documents(documents)
    store.update_embeddings(retriever, update_existing_embeddings=False)


if __name__ == "__main__":
    current = os.path.dirname(os.path.realpath(__file__))

    data = read_file(f'{current}/Viina_i_myr_Tom_1-2_vyd_1952.txt')
    chapters = extract_xml_chapters(''.join(data))
    data = read_file(f'{current}/Viina_i_myr_Tom_3-4_vyd_1952.txt')
    chapters.extend(extract_xml_chapters(''.join(data)))
    preprocess(chapters)
