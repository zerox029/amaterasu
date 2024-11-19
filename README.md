# Amaterasu

Amaterasu is a japanese tokenizer based on an LSTM network. The model architecture is heavily inspired by Yoshiaki Kitagawa and Mamoru Komachi's paper [Long Short-Term Memory for Japanese Word Segmentation](https://aclanthology.org/Y18-1033.pdf), Taishi Ikeda, Hiroyuki Shindo and Yuuji Matsumoto's paper [辞書情報と単語分散表現を組み込んだ
リカレントニューラルネットワークによる日本語単語分割](https://www.anlp.jp/proceedings/annual_meeting/2017/pdf_dir/B6-2.pdf) and Geewook Kim, Kazuki Fukui, Hidetoshi Shimodaira's 2018 paper [Segmentation-free Compositional n-gram Embedding](https://aclanthology.org/N19-1324.pdf). Additional information taken from Takeshi Arabiki's blog post
[日本語形態素解析の裏側を覗く！MeCab はどのように形態素解析しているか](https://techlife.cookpad.com/entry/2016/05/11/170000) and Taku Kudou's book [形態素解析の理論と実装](https://www.amazon.co.jp/%E5%BD%A2%E6%85%8B%E7%B4%A0%E8%A7%A3%E6%9E%90%E3%81%AE%E7%90%86%E8%AB%96%E3%81%A8%E5%AE%9F%E8%A3%85-%E5%AE%9F%E8%B7%B5%E3%83%BB%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%82%B7%E3%83%AA%E3%83%BC%E3%82%BA-%E5%B7%A5%E8%97%A4-%E6%8B%93/dp/4764905779).

## Getting Started
To run Amaterasu, you will first need to generate embeddings for n-grams of size
1 through 5. To do so, please refer to the SCNE submodule present in `embeddings/`.
They can be trained on any corpus but if you wish to train them on Wikipedia like I did,
a submodule to do so easily is also present.

## Requirements
- Python 3.12 or higher
- NGram embeddings (refer to the above)

## Todo
- [ ] Implement CRF layer
- [x] Bucket batching
- [x] Add support for external corpus
- [ ] Experiment with moving target character to different places within the ngram
- 