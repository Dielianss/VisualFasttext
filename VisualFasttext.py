import numpy as np
import fasttext

EOS = "</s>"
BOW = "<"
EOW = ">"


class VisualFasttext:
    def __init__(self, model_path: str, print_specifications=False):
        self.model = fasttext.load_model(model_path)
        self.input_matrix = self.model.get_input_matrix()
        self.output_matrix = self.model.get_output_matrix()
        self.labels = self.model.get_labels()
        self.args = self.model.f.getArgs()

        self.r = np.uint64(116049371)
        self.bucket = np.uint64(self.args.bucket)
        self.nwords = np.uint64(len(self.model.get_words()))
        self.len_ft_words = 0

        if print_specifications:
            self.get_model_specifications()

    def get_model_specifications(self):
        """
        get the specifications of the given fasttext model.
        """
        print("\nThe specifications of the given fasttext model are listed below:")
        print("  dim of embedding: ", self.args.dim)
        print("  window size: ", self.args.ws)
        print("  epoch: ", self.args.epoch)
        print("  minCount: ", self.args.minCount)
        print("  wordNgrams: ", self.args.wordNgrams)
        print("  model: ", self.args.model)
        print("  bucket: ", self.args.bucket)
        print("  minn: ", self.args.minn)
        print("  maxn: ", self.args.maxn)
        print("  learning rate: ", self.args.lr)
        print("  number of words: ", len(self.model.get_words()))
        print("  number of labels: ", len(self.model.get_labels()))
        print()

    def tokenize(self, input_str: str):
        return self.model.f.tokenize(input_str)

    def encode_raw_words(self, raw_words: list):
        """
        encode raw words in fasttext way (replicating function .getLine() in /src/dictionary.cc)
        """
        raw_words.append(EOS)
        ft_words = []  # all the "effective" fasttext words, excluding out-of-vocabulary words
        ft_word_ids = []  # ids of all the "effective" fasttext words

        ft_word_hashes = []  # hash values of all the raw words (including out-of-vocabulary words) and their subwords.

        for raw_word in raw_words:
            word_id = self.model.get_word_id(raw_word)
            # .get_hash() returns a uint_32 number, we need to convert it to a int32 number
            uhash = np.int32(self.model.get_hash(raw_word))
            ft_word_hashes.append(uhash)  # each raw word will always have a hash value even it is out-of-vocabulary
            self.add_subwords(raw_word, word_id, ft_word_ids, ft_words)

            if raw_word == EOS:
                break

        self.add_word_ngrams(ft_words, ft_word_ids, ft_word_hashes, raw_words)
        self.len_ft_words = len(ft_words)

        return ft_words, ft_word_ids

    def add_word_ngrams(self, ft_words: list, ft_word_ids: list, ft_word_hashes: list, raw_words: list):
        """
        add wordNgrams (replicating function .addWordNgrams() in /src/dictionary.cc)
        according to its source code, Fasttext will add wordNgrams based on ft_word_hashes rather than
        ft_word_ids, which means when wordNgrams is enabled, actually each raw word (even for a out-of-vocabulary one)
        will make a contribution to the final prediction, since it will always have a hash value.
        """
        if self.args.wordNgrams <= 1:
            return ft_word_ids

        len_word_hashes = len(ft_word_hashes)

        for i in range(0, len_word_hashes):
            h = np.uint64(ft_word_hashes[i])
            token_ngram = raw_words[i]

            for j in range(i+1, i+self.args.wordNgrams):
                if j >= len_word_hashes:
                    break

                h = np.add(h * self.r, np.uint64(ft_word_hashes[j]))
                token_ngram += raw_words[j]

                if h > 0:
                    ft_word_ids.append(np.add(self.nwords, np.mod(h, self.bucket)))
                    ft_words.append(token_ngram)

    def compute_subwords(self, raw_word: str, ft_word_ids: list, ft_words: list):
        """
        get the subwords of a out-of-vocabulary raw word (replicating function computeSubwords in /src/dictionary.cc)
        """
        len_raw_word = len(raw_word)
        for i in range(len_raw_word):
            subword = ""

            j, n = i, 1
            while j < len_raw_word and n <= self.args.maxn:
                subword += raw_word[j]
                j += 1
                if n >= self.args.minn and not (n == 1 & (i == 0 or j == len_raw_word)):
                    h = self.model.get_hash(subword)

                    if h >= 0:
                        ft_word_ids.append(np.add(self.nwords, np.mod(np.uint32(h), self.bucket)))
                        ft_words.append(subword)

                n += 1

    def add_subwords(self, raw_word: str, raw_word_id: int, ft_word_ids: list, ft_words: list):
        """
        add subwords for a given word (replicating function addSubwords in /src/dictionary.cc)
        """
        if raw_word_id < 0:  # the word is out of vocabulary
            if raw_word != EOS:  # compute subwords for all the raw words except EOS
                self.compute_subwords("<{}>".format(raw_word), ft_word_ids, ft_words)
        else:
            if self.args.maxn <= 0:  # function "subwords" of the given fasttext model is disabled
                ft_word_ids.append(raw_word_id)
                ft_words.append(raw_word)
            else:
                subwords, subword_ids = self.model.get_subwords(raw_word)
                ft_word_ids.extend(subword_ids)
                ft_words.extend(subwords)

    def softmax(self, input_data):
        max_score = np.max(input_data)
        output = np.exp(input_data - max_score)
        z = np.sum(output)

        return np.divide(output, z)

    def calc_hidden(self, ft_words_statistics: dict):
        for _, value in ft_words_statistics.items():
            embedding = self.input_matrix[value['id']]
            hidden = embedding @ self.output_matrix.T
            scores = np.divide(hidden, self.len_ft_words)
            value['scores'] = scores

        return ft_words_statistics

    def calc_prediction(self, ft_words_statistics: dict):
        """
        get the most possible prediction and the contributions of each ft words to this prediction
        """
        ft_word_logits_list = []
        for key, value in ft_words_statistics.items():
            w, s = value['count'], value['scores']
            ft_word_logits_list.append(w * s)

        ft_word_logits = np.array(ft_word_logits_list)
        sum_logits = np.sum(ft_word_logits, axis=0)
        final_scores = self.softmax(sum_logits)
        sorted_scores_idx = np.argsort(final_scores)[-1]

        target_component = ft_word_logits[:, sorted_scores_idx]
        predicted_score = final_scores[sorted_scores_idx]
        predicted_label = self.labels[sorted_scores_idx]

        contributions = self.analysis_feature(ft_words_statistics, target_component)

        return predicted_label, predicted_score, contributions

    def get_topK(self, sentence_vector, k=1):
        """
        get the top K predictions
        """
        logits = sentence_vector @ self.output_matrix.T
        final_scores = self.softmax(logits)
        sorted_scores_idx = np.argsort(final_scores)[0][-k:]

        sorted_scores = final_scores[0][sorted_scores_idx]
        labels = np.array(self.labels)
        predicted_labels = labels[sorted_scores_idx]

        return sorted_scores, predicted_labels

    def analysis_feature(self, ft_words_statistics: dict, component):
        """
        compute contribution scores for each "effective" fasttext words (words in ft_words) and sort them in descending
        order
        """
        softmax_component = self.softmax(component)
        sorted_tokens = np.argsort(softmax_component, axis=0)[::-1]  # descending order
        tokens = np.array(list(ft_words_statistics.keys()))
        contributions = dict(zip(tokens[sorted_tokens], softmax_component[sorted_tokens]))

        return contributions

    def get_ft_words_statistics(self, ft_word_ids: list, ft_words: list):
        """
        get the statistics of ft_words, including their frequencies and word_ids.
        """
        ft_words_statistics = {}
        for word_id, word in zip(ft_word_ids, ft_words):
            if word not in ft_words_statistics:
                ft_words_statistics[word] = {'id': word_id, 'count': 0}
            ft_words_statistics[word]['count'] += 1

        return ft_words_statistics

    def get_sentence_vector(self, input_str: str):
        """
        replicate function .get_sentence_vector() of FastText's Python Wrapper
        """
        _, ft_word_ids = self.encode_raw_words(self.tokenize(input_str))
        sentence_vector = np.zeros((1, self.args.dim), dtype='float')

        for word_id in ft_word_ids:
            embedding = self.input_matrix[word_id]
            sentence_vector += embedding

        return np.divide(sentence_vector, self.len_ft_words)

    def predict(self, input_str: str, k=1):
        """
        replicate function .predict() of FastText's Python Wrapper
        """
        sentence_vector = self.get_sentence_vector(input_str)
        prediction = self.get_topK(sentence_vector, k)

        return prediction

    def predict_plus(self, input_str: str):
        """
        given the string, get the prediction of fasttext model, then compute and sort the contributions of each ft_words
        to the final prediction in descending order.
        """
        ft_words, ft_word_ids = self.encode_raw_words(self.tokenize(input_str))
        ft_words_statistics = self.get_ft_words_statistics(ft_word_ids, ft_words)
        hidden_statistics = self.calc_hidden(ft_words_statistics)
        predicted_label, predicted_score, contributions = self.calc_prediction(hidden_statistics)

        return predicted_label, predicted_score, contributions
