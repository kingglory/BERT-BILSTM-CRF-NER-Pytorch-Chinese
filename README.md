#### BERT-BiLSTM-CRF模型

##### 【简介】使用谷歌的BERT模型在BiLSTM-CRF模型上进行预训练用于中文命名实体识别的pytorch代码

##### 项目结构

```
bert_bilstm_crf_ner_pytorch
    torch_ner
        bert-base-chinese           --- 预训练模型
        data                        --- 放置训练所需数据
        output                      --- 项目输出，包含模型、向量表示、日志信息等
        source                      --- 源代码
            config.py               --- 项目配置，模型参数
            conlleval.py            --- 模型验证
            logger.py               --- 项目日志配置
            models.py               --- bert_bilstm_crf的torch实现
            main.py                 --- 模型训练
            processor.py            --- 数据预处理
            predict.py              --- 模型预测
            utils.py                --- 工具包
```

##### 数据预处理

输入数据格式请处理成BIO格式，放置在data/old_data目录下，格式如下:

```
在 O
广 B-LOC
西 I-LOC
壮 I-LOC
族 I-LOC
自 I-LOC
治 I-LOC
区 I-LOC
柳 I-LOC
州 I-LOC
市 I-LOC
柳 I-LOC
南 I-LOC
区 I-LOC
航 I-LOC
鹰 I-LOC
大 I-LOC
道 I-LOC
租 O
房 O
住 O
的 O
张 B-NAME
之 I-NAME
三 I-NAME
是 O
什 O
么 O
人 O
？ O

辽 B-LOC
宁 I-LOC
省 I-LOC
海 I-LOC
城 I-LOC
市 I-LOC
的 O
李 B-NAME
丘 I-NAME
闪 I-NAME
```

##### 使用方法

- 修改项目配置
- 训练
  
  ```
  train()
  ```
- 预测
  
  ```
  predict("xxx")
  ```
  
  具体可见`train.py`、`predict.py`

##### 关于BERT-BiLSTM-CRF

```
class BERT_BiLSTM_CRF(BertPreTrainedModel):
    """
        BERT:
            outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
            # torch.Size([batch_size,seq_len,hidden_size]) --- [6,128,768]
            sequence_output = outputs[0]

            Inputs:
                input_ids:      torch.Size([batch_size,seq_len]), 代表输入实例的tensor张量
                token_type_ids: torch.Size([batch_size,seq_len]), 一个实例可以含有两个句子,相当于标记
                attention_mask: torch.Size([batch_size,seq_len]), 指定对哪些词进行self-Attention操作
            Out:
                sequence_output: torch.Size([batch_size,seq_len,hidden_size]), 输出序列
                pooled_output:   torch.Size([batch_size,hidden_size]), 对输出序列进行pool操作的结果
                (hidden_states): tuple, 13*torch.Size([batch_size,seq_len,hidden_size]), 隐藏层状态，取决于config的output_hidden_states
                (attentions):    tuple, 12*torch.Size([batch_size, 12, seq_len, seq_len]), 注意力层，取决于config中的output_attentions

        BiLSTM:
            # input_size:768, hidden_size=128
            self.birnn = nn.LSTM(input_size=config.hidden_size, hidden_size=rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            Args:
                input_size:    输入数据的特征维数
                hidden_size:   LSTM中隐层的维度
                num_layers:    循环神经网络的层数
                bias:          用不用偏置，default=True
                batch_first:   通常我们输入的数据shape=(batch_size,seq_length,input_size),而batch_first默认是False,需要将batch_size与seq_length调换
                dropout:       默认是0，代表不用dropout
                bidirectional: 默认是false，代表不用双向LSTM

            # [6,128,768] --> [6,128,256]
            sequence_output, _ = self.birnn(sequence_output)
            Inputs:
                input:     shape=(seq_length,batch_size,input_size)的张量
                (h_0,c_0): h_0的shape=(num_layers*num_directions,batch,hidden_size)的张量，它包含了在当前这个batch_size中每个句子的初始隐藏状态，
                           num_layers就是LSTM的层数，如果bidirectional=True,num_directions=2,否则就是１，表示只有一个方向，c_0和h_0的形状相同，
                           它包含的是在当前这个batch_size中的每个句子的初始状态，h_0、c_0如果不提供，那么默认是０
            OutPuts:
                output:    shape=(seq_length,batch_size,num_directions*hidden_size), 它包含的LSTM的最后一层的输出特征(h_t),t是batch_size中每个句子的长度
                (h_n,c_n): h_n.shape=(num_directions * num_layers,batch,hidden_size), c_n与H_n形状相同, h_n包含的是句子的最后一个单词的隐藏状态，
                           c_n包含的是句子的最后一个单词的细胞状态，所以它们都与句子的长度seq_length无关;output[-1]与h_n是相等的，因为output[-1]包含的正是batch_size
                           个句子中每一个句子的最后一个单词的隐藏状态，注意LSTM中的隐藏状态其实就是输出，cell state细胞状态才是LSTM中一直隐藏的，记录着信息

        全连接层:
            self.hidden2tag = nn.Linear(out_dim, config.num_labels)
            Args:
                in_features:  输入的二维张量的大小，即输入的[batch_size, size]中的size
                out_features: 输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数

            释义:
                从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量

        CRF:
            self.crf = CRF(config.num_labels, batch_first=True)
            Args:
                num_tags:    Number of tags.
                batch_first: Whether the first dimension corresponds to the size of a minibatch.
    """

    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.need_birnn = need_birnn

        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2

        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, tags, token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        return loss

    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return self.crf.decode(emissions, attention_mask.byte())
```

#### 验证指标

![](C:\Users\daobi\AppData\Roaming\marktext\images\2022-12-19-16-15-20-image.png)

##### 参考文章

- [从Word Embedding到Bert模型——自然语言处理预训练技术发展史](https://mp.weixin.qq.com/s?__biz=Mzg4NDQwNTI0OQ==&mid=2247523426&idx=2&sn=e608ddbb23a44031a11292f48670fa57)
- [一文读懂BERT(原理篇)](https://blog.csdn.net/jiaowoshouzi/article/details/89073944)
- [BERT — transformers 4.5.0](https://huggingface.co/transformers/model_doc/bert.html)
- [Pytorch-Bert预训练模型的使用](https://www.cnblogs.com/douzujun/p/13572694.html)
- [通俗易懂理解——BiLSTM](https://zhuanlan.zhihu.com/p/40119926)  
- [详解BiLSTM及代码实现](https://zhuanlan.zhihu.com/p/47802053)  
- [torch.nn.LSTM()详解](https://blog.csdn.net/m0_45478865/article/details/104455978)
- [LSTM细节分析理解（pytorch版）](https://zhuanlan.zhihu.com/p/79064602)
- [LSTM+CRF 解析（原理篇）](https://zhuanlan.zhihu.com/p/97829287)  
- [crf模型原理及解释](https://www.jianshu.com/p/e608cdfdc174)
