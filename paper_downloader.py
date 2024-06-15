import argparse
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


class TypedArgs(argparse.Namespace):
    url: str
    save_dir: str

    @staticmethod
    def from_argparse(args: argparse.Namespace) -> "TypedArgs":
        return TypedArgs(**vars(args))


class DownloaderBase:
    def download(self, url: str, save_dir: str):
        raise NotImplementedError()

class ACLAnthologyDownloader(DownloaderBase):
    def download(self, url: str, save_dir: str = "./"):
        req = requests.get(url)
        soup = BeautifulSoup(req.text, "html.parser")
        title = soup.find(id="title").text.replace(" ", "_")
        pdf_url = url.replace("abs", "pdf")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(title, pdf_url, save_dir)
        with open(save_dir / f"{title}.pdf", "wb") as f:
            f.write(requests.get(pdf_url).content)

class ArXivDownloader(DownloaderBase):
    def download(self, url: str, save_dir: str = "./"):
        req = requests.get(url)
        soup = BeautifulSoup(req.text, "html.parser")
        title = soup.find("h1", class_="title mathjax").text.replace(" ", "_")
        pdf_url = url.replace("abs", "pdf")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(title, pdf_url, save_dir)
        with open(save_dir / f"{title}.pdf", "wb") as f:
            f.write(requests.get(pdf_url).content)


def main(args: TypedArgs):
    url = args.url
    if url[-1] == "/":
        url = url[:-1]
    parsed_url = urlparse(url)
    if parsed_url.netloc == "aclanthology.org":
        downloader = ACLAnthologyDownloader()
    elif parsed_url.netloc == "arxiv.org":
        downloader = ArXivDownloader()
    else:
        raise NotImplementedError(f"Unsupported URL: {url}")

    downloader.download(args.url, args.save_dir)


parser = argparse.ArgumentParser()
parser.add_argument("url", type=str)
parser.add_argument("--save_dir", type=str, default="./pdf/")
args = TypedArgs.from_argparse(parser.parse_args())
main(args)
