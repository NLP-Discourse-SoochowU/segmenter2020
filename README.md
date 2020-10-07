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
   Since we use multi-head attention in this work for robustness, we use the
   following loss objective to encourage the divergence between each two attention
   heads.
```
<div align=center>
<img src="https://github.com/NLP-Discourse-SoochowU/segmenter2020/blob/master/data/img/model.png" width="700" alt="eg"/>
<br/>
<img src="https://github.com/NLP-Discourse-SoochowU/segmenter2020/blob/master/data/img/loss.png" width="360" alt="eg"/>
<br/>
</div>

<b>-- Example Analysis</b>
```
   To study the correlation between the EDU segmentation process and the syntactic
   information we use, we give another analysis about the randomly selected
   examples in the Figure below. In dependency structure, a fake root is usually added
   and only one word is the dependent of the root, which we refer to as the root-dep
   unit (e.g., the word "have" in Figure (a)). Intuitively, we draw partial dependency
   structure between EDU boundaries and root-dep units for the two examples
   respectively. And the partial dependency structures in both examples reveal an
   interesting language phenomenon that those words identifying EDU boundaries
   are direct dependents of root-dep units. Scrupulously, we further display the
   proportion of EDU boundaries related to root-dep units in Table 6, and the
   results show that this language phenomenon is common in both corpora. Under
   the conduction of explicit dependency structures, those text units serving as
   dependents of root-dep units are well equipped with "hints" for EDU boundary
   determination. Hence, we have reason to believe that the refining method we
   use is stable and useful for RST-style discourse segmentation for languages like
   English and Chinese.
```

<div align=center>
<img src="https://github.com/NLP-Discourse-SoochowU/segmenter2020/blob/master/data/img/ana.png" width="700" alt="eg"/>
<br/>
</div>

<b>-- Required Packages</b>
```
   Coming soon
```

<b>-- Usage</b>
```
   Coming soon.
```

<b>-- Reference</b>

   Please read the following paper for more technical details

   Longyin Zhang, Fang Kong and Guodong Zhou, Syntax-Guided Sequence to Sequence Modeling for Discourse Segmentation.
   Conference paper of NLPCC2020.

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
