import unicodedata


def load_vocab(vocab_path, encoding='utf-8', simplified=False, startwith=None):
    """从bert的词典文件中读取词典"""
    token_dict = {}
    with open(vocab_path, encoding=encoding) as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    if simplified:  # 过滤冗余部分token
        new_token_dict, keep_tokens = {}, []
        startwith = startwith or []
        for t in startwith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

        for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
            if t not in new_token_dict:
                keep = True
                if len(t) > 1:
                    for c in (t[2:] if t[:2] == '##' else t):
                        if is_cjk_char(c) or is_punctuation(c):
                            keep = False
                            break
                if keep:
                    new_token_dict[t] = len(new_token_dict)
                    keep_tokens.append(token_dict[t])

        return new_token_dict, keep_tokens
    else:
        return token_dict


class Tokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 vocab_path,
                 do_lower_case=True,
                 token_start='[CLS]',
                 token_end='[SEP]', ):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.vocab = load_vocab(vocab_path)
        self.token_start = token_start
        self.token_end = token_end
        self.token_pad = self.vocab['[PAD]']
        self.token_unk = self.vocab['[UNK]']
        self.token_mask = self.vocab['[MASK]']

    def tokenize(self, text):
        text = convert_to_unicode(text)
        clean_text = ''
        for ch in text:
            if is_cjk_char(ch) or is_punctuation(ch):
                clean_text += ' ' + ch + ' '
            elif is_whitespace(ch):
                clean_text += ' '
            elif is_invalid_char(ch):
                continue
            else:
                clean_text += ch

        tokens = []
        for word in clean_text.strip().split():
            tokens.extend(self._word_piece_tokenize(word))

        return tokens

    def _word_piece_tokenize(self, word):
        """word内分成subword
        """
        if word in self.vocab:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self.vocab:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop

        return tokens

    def encode(self,
               first_text,
               second_text=None,
               max_length=None,
               first_length=None,
               second_length=None):
        """输出文本对应token id和segment id
        如果传入first_length，则强行padding第一个句子到指定长度；
        同理，如果传入second_length，则强行padding第二个句子到指定长度。
        """
        if is_string(first_text):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif is_string(second_text):
            idx = int(bool(self.token_start))
            second_tokens = self.tokenize(second_text)[idx:]
        else:
            second_tokens = second_text

        if max_length is not None:
            self.truncate_sequence(max_length, first_tokens, second_tokens, -2)

        first_token_ids = self.tokens_to_ids(first_tokens)
        if first_length is not None:
            first_token_ids = first_token_ids[:first_length]
            first_token_ids.extend([self.token_pad] *
                                   (first_length - len(first_token_ids)))
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            second_token_ids = self.tokens_to_ids(second_tokens)
            if second_length is not None:
                second_token_ids = second_token_ids[:second_length]
                second_token_ids.extend(
                    [self.token_pad] *
                    (second_length - len(second_token_ids)))
            second_segment_ids = [1] * len(second_token_ids)

            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    def truncate_sequence(self,
                          max_length,
                          first_sequence,
                          second_sequence=None,
                          pop_index=-1):
        """截断总长度
        """
        if second_sequence is None:
            second_sequence = []

        while True:
            total_length = len(first_sequence) + len(second_sequence)
            if total_length <= max_length:
                break
            elif len(first_sequence) > len(second_sequence):
                first_sequence.pop(pop_index)
            else:
                second_sequence.pop(pop_index)

    def tokens_to_ids(self, tokens):
        """token序列转换为对应的id序列
        """
        return [self.vocab[token] for token in tokens]


def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, str)


def convert_to_unicode(text):
    """把text转换成unicode编码的形式"""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def is_cjk_char(ch):
    """判断是否是中日韩的文字
    https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    """
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or \
           0x3400 <= code <= 0x4DBF or \
           0x20000 <= code <= 0x2A6DF or \
           0x2A700 <= code <= 0x2B73F or \
           0x2B740 <= code <= 0x2B81F or \
           0x2B820 <= code <= 0x2CEAF or \
           0xF900 <= code <= 0xFAFF or \
           0x2F800 <= code <= 0x2FA1F


def is_punctuation(ch):
    """判断是否是标点符号"""
    code = ord(ch)
    return 33 <= code <= 47 or \
           58 <= code <= 64 or \
           91 <= code <= 96 or \
           123 <= code <= 126 or \
           unicodedata.category(ch).startswith('P')


def is_invalid_char(ch):
    """判断是否是无效的字符和空格"""
    # 转成unicode编码
    cp = ord(ch)
    # 0是NULL 0xfffd是可替换字符，表示的是未知字符
    return cp == 0 or cp == 0xfffd or is_control(ch)


def is_control(ch):
    """判断'char'是否是Control or Format"""
    if ch == "\t" or ch == "\n" or ch == "\r":
        return False
    cat = unicodedata.category(ch)
    # Cc表示Control Cf表示Format
    if cat in ("Cc", "Cf"):
        return True
    return False


def is_whitespace(ch):
    """判断是否是空格"""
    if ch == " " or ch == "\t" or ch == "\n" or ch == "\r":
        return True
    # unicodedata是unicode数据集
    cat = unicodedata.category(ch)
    # Zs表示的是空格
    if cat == "Zs":
        return True
    return False


if __name__ == '__main__':
    tokenize = Tokenizer('chinese_L-12_H-768_A-12/vocab.txt')
    a = tokenize.encode('奥术\t大hit师&(*%^&大所, 大所!!!  多')
    print(a)
