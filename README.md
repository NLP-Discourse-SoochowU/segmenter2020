## Syntax_guided EDU Segmentation Model

<b>-- General Information</b>
```
   [Abstract] Previous studies on RST-style discourse segmentation have
   achieved impressive results. However, recent neural works either require a
   complex joint training process or heavily rely on powerful pre-trained word
   vectors. Under this condition, a simpler but more robust segmentation
   method is needed. In this work, we take a deeper look into intra-sentence
   dependencies to investigate if the syntax information is totally useless,
   or to what extent it can help improve the discourse segmentation perfor-
   mance. To achieve this, we propose a sequence-to-sequence model along
   with a GCN based encoder to well utilize intra-sentence dependencies
   and a multi-head biaffine attention based decoder to predict EDU bound-
   aries. Experimental results on two benchmark corpora show that the
   syntax information we use is significantly useful and the resulting model
   is competitive when compared with the state-of-the-art.
```

<b>-- Model Architecture</b>
```
   The figure below illustrates the proposed approach.
```
<div align=center>
<img src="https://github.com/NLP-Discourse-SoochowU/segmenter2020/blob/master/data/img/model.png" width="700" alt="eg"/>
<br/>
</div>
<b>--  </b>
```
   Since we use multi-head attention in this work for robustness, we use the
   following loss objective to encourage the divergence between each two attention
   heads.
```
<div align=center>
<img src="https://github.com/NLP-Discourse-SoochowU/segmenter2020/blob/master/data/img/loss.png" width="360" alt="eg"/>
<br/>
</div>
<b>--  </b>
```
   Here, we display two segmentation results for analysis.
```
<div align=center>
<img src="https://github.com/NLP-Discourse-SoochowU/segmenter2020/blob/master/data/img/ana.png" width="700" alt="eg"/>
<br/>
</div>

<b>-- Required Packages</b>
```
   torch==0.4.0
   numpy==1.14.1
   nltk==3.3
   stanfordcorenlp==3.9.1.1
```

<b>-- Train Your Own RST Parser</b>
```
    Run main.py

```

<b>-- RST Parsing with Raw Documents</b>
```
   1. Prepare your raw documents in data/raw_txt in the format of *.out
   2. Run the Stanford CoreNLP with the given bash script corpus_rst.sh
      using the command "./corpus_rst.sh ". Of course, if you use other
      models for EDU segmentation then you do not need to perform the
      action in step 2.
   3. Run parser.py to parse these raw documents into objects of rst_tree
      class (Wrap them into trees).
      - segmentation (or you can use your own EDU segmenter)
      - wrap them into trees, saved in "data/trees_parsed/trees_list.pkl"
   4. Run drawer.py to draw those trees out by NLTK
   Note: We did not provide parser codes and it can be easily implemented referring to our previous project.
```
[rst_dp2018](https://github.com/NLP-Discourse-SoochowU/rst_dp2018)

<b>-- Reference</b>

   Please read the following paper for more technical details

   [Longyin Zhang, Xin Tan, Fang Kong and Guodong Zhou, A Recursive Information Flow Gated Model for RST-Style Text-Level Discourse Parsing.](http://tcci.ccf.org.cn/conference/2019/papers/119.pdf)

<b>-- Developer</b>
```
  Longyin Zhang
  Natural Language Processing Lab, School of Computer Science and Technology, Soochow University, China
  mail to: zzlynx@outlook.com, lyzhang9@stu.suda.edu.cn

```

<b>-- License</b>
```
   Copyright (c) 2019, Soochow University NLP research group. All rights reserved.
   Redistribution and use in source and binary forms, with or without modification, are permitted provided that
   the following conditions are met:
   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
      following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.
```
