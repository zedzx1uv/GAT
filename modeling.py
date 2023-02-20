import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel 

class MLPLayer(nn.Module):
	"""
	Head for getting sentence representations over RoBERTa/BERT's CLS representation.
	"""

	def __init__(self, hidden_size=768):
		super().__init__()
		self.dense = nn.Linear(hidden_size, hidden_size)
		self.activation = nn.Tanh()

	def forward(self, features, **kwargs):
		x = self.dense(features)
		x = self.activation(x)

		return x

class Similarity(nn.Module):
	"""
	Dot product or cosine similarity
	"""

	def __init__(self, temp):
		super().__init__()
		self.temp = temp
		self.cos = nn.CosineSimilarity(dim=-1)

	def forward(self, x, y):
		return self.cos(x, y) / self.temp



class BertForCL(nn.Module):
	def __init__(self, model, base_model='bert'):
		super().__init__()
		self.bert = getattr(model, base_model)
		self.mlp = MLPLayer(hidden_size=768)
		self.sim = Similarity(temp=1e-8)
		self.mlp.cuda()

	def cl_forward(self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
		mlm_input_ids=None,
		mlm_labels=None,
	):
		# ori_input_ids = input_ids
		# batch_size = input_ids.size(0)
		# # Number of sentences in one instance
		# # 2: pair instance; 3: pair instance with a hard negative
		# num_sent = input_ids.size(1)
		# print(num_sent)
		# return 
		# mlm_outputs = None
		# # Flatten input for encoding
		# input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
		# attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
		# if token_type_ids is not None:
		# 	token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
		# Get raw embeddings
		outputs = self.bert(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=False,
			return_dict=True,
		)
		# Pooling
		pooler_output = outputs 
		# print(pooler_output.shape)
		# pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
		pooler_output = self.mlp(pooler_output[0])
		# print(pooler_output[:,0].shape)
		# z1, z2 = pooler_output[:,0], pooler_output[:,1]
		# cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
		# labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
		# loss_fct = nn.CrossEntropyLoss()

		# loss = loss_fct(cos_sim, labels)

		# return SequenceClassifierOutput(
		# 	loss=loss,
		# 	logits=cos_sim,
		# 	hidden_states=outputs.hidden_states,
		# 	attentions=outputs.attentions,
		# )
		return pooler_output[:,0]