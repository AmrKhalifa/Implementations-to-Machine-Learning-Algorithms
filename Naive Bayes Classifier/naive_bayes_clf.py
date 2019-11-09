import math 
class NaiveBayes():
	
	def __init__(self, MAP = False, regularize = False):
		
		self.MAP = MAP
		self.regularize = regularize
		self.word_dic = None  

	def fit(self, corpus):
		self.corpus = corpus
		self.word_dic = self.process_file(self.corpus)


	def _process_text(self, text):
		text = text.replace(',', '')
		text = text.replace('.', '')
		text = text.replace('!', '')
		text = text.replace('?', '')
		text = text.replace('-', '')

		text = text.lower()

		return text 

	def _process_file(self, file): 
		word_dic = {}
		label_counts = {}
		with open(file, 'r', encoding='utf8') as f: 
			for line in f: 
				label, text = line.split("\t")
				try:
					label_counts[label]
					label_counts[label] += 1
				except:
					label_counts[label] = 1
				text = self._process_text(text)
				text_list = text.split()
				try:
					word_dic[label] is True
					for word in text_list:  
						try:
							word_dic[label][word] is True
							word_dic[label][word] += 1
						except:
							word_dic[label][word] = 1 
				except:
					word_dic[label] = {}

		return word_dic

	def _get_word_counts(self, word):

		word_counts = {}
		for label in self.word_dic.keys():
			try: 
				word_counts[label] = self.word_dic[label][word]
			except:
				 word_counts[label] = 0

		return word_counts

	def _get_no_words_per_class(self):

		label_counts = {}
		for label in self.word_dic.keys():
			label_counts[label] = sum(self.word_dic[label].values())
		
		return label_counts


	def _compute_log(self, a, b):
		try: 
			value = math.log(a/b)
		except:
			value = -10000000

		return value


	def _compute_log_text(self, text):

		per_class_log_text = {}

		for word in text.split():
			no_words_per_class = self._get_no_words_per_class(self.word_dic)
			word_count = self._get_word_counts(word, self.word_dic)
 
			for label in self.word_dic.keys():
				try:
					per_class_log_text[label] += self._compute_log(word_count[label], no_words_per_class[label])
				except:	
					per_class_log_text[label] = self._compute_log(word_count[label], no_words_per_class[label])

		return per_class_log_text

	def classify(self, text, MAP = False):
		text = self._process_text(text)
		per_class_log_text = self._compute_log_text(text, self.word_dic)
		return max(per_class_log_text.items(), key = lambda x: x[1])
		pass 


file = 'data/train1.txt'
classifier = NaiveBayes()
classifier.fit(file)
print(classifier.classify("there here now Here now", False))



