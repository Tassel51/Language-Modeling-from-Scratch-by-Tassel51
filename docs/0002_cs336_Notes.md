课程视频参考来源：【高质量配音】2025年 斯坦福 CS336 从头构建大语音模型 ｜ 中文配音+英文原声双版本【已完结】https://www.bilibili.com/video/BV1APuUz7Erx?vd_source=6ff8ea83415789221ed18af6aea45d13

<mark>**Stanford CS336 Language Modeling from Scratch | Spring 2025**</mark>

>组织主页：https://github.com/stanford-cs336
>
>2025 春季课程主页：https://stanford-cs336.github.io/spring2025/


<br>

# Part1:课程概述与分词器
## 1.1 课程概述
1. 该课程主要为构建小规模的语言模型，但是可能没有代表性，因为参数多的时候mlp层会完全占主导，随着训练计算量的增加，模型准确率会大幅度增加，模型会涌现更多的智能，在小规模下有效的架构和数据集在大规模下就可能不适用了
2. efficent is important, design efficient arcitectures
3. 数据处理转换过程1

   
## 1.2 分词器
1. implemenet BPE tokenizer(感谢[相关博主的大作业](https://www.heywhale.com/mw/project/689709e023583639fc675b5c))
2. 分词器的类型
   -  character-based tokenizer 有的词汇出现的多有的少，效率不高，后面会有id=12222这种很大的数值，但是出现的字数少就效率不高
   -  byte-based tokenizer 转换成字节，utf-8编码一个字符可能会有4个字节表示，同时字节的类型总共也只有256个，这样词汇表就很小了，但是序列太长了，效率很低，压缩比很高
   -  word-based tokenizer 把字符串分割为一系列子串，给每个片段一个整数编号，但是新出现的词会标记成未知字符会出问题
   -  **Byte Pair Encoding (BPE)**  used by GPT-2，不预设任何分词规则而是在原始文本上训练分词器，常见的用一个词元表示，罕见的用多个表示
      -  首先，先转换成整数序列（字节表示），然后统计每个整数出现的次数，然后按照出现次数排序，然后选择出现次数最高的两个整数进行合并，这样索引列表就在变短，压缩效率就在提升；BPE分词器也可以用于解码。**关键的是，他会看语料统计，做出合理决策，指导我们如何动态最优的分配词表来表示字符序列**

<br>

# Part2：PyTorch与资源管理
1. 一般每个数字都是默认float32的表示方法，但是float16的表示方法可以节省一半的内存，但是不适合处理几小数或者极大数，大模型可能会出问题（像某些参数需要随时间累计的需要更高精度的我们一般会用这个float32）
2. bfloat16是google提出的，更看重动态值范围而不是精度，有float16的内存但是有32的动态范围，但是分辨率却是会差一点
3. einops库，可以方便的进行张量操作，为所有维度命名而不是依赖数字索引


<br>

# Part3：模型架构与超参数
## 3.1 模型架构方面
1. Pre-vs-post norm,the data 现在一般采用前置归一化，前置归一化会使训练变得更加稳定，现在好多都是前后都加归一化；为什么在残差连接不好？层归一化放在中间，可能会影响梯度连接的特性。
2. RMSNorm：为什么用这种归一化？这种归一化的计算量更小，主要影响的时间会在计算矩阵时从内存搬运的速度，影响的时间比较多；Memory Movement is all you need 

<br>

# Part4：混合专家模型