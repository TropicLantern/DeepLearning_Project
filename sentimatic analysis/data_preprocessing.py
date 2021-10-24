
# clear punctuation
def without_punctuation(src, des):   # src是预处理的数据地址,des是处理完数据放置的位置
    txt1 = open(src, 'r', encoding="UTF-8").read()
    punctuation = "`~!@#$%^&*()_-=/<>,.?:;[]|\\{}"
    for ch in punctuation:
        txt1 = txt1.replace(ch, '')
    with open(des, 'w', encoding="UTF-8") as f:
        f.write(txt1)
        
        
without_punctuation("E:/pycharm_project/data/train-1.txt", "E:/pycharm_project/data/train-2.txt")

# clear StopWords
def Clear_StopWords(src, des):
    # stop_words
    txt2 = open(src, 'r', encoding="UTF-8").read()
    stopwords = [line.strip() for line in open('E:/pycharm_project/data/StopWords.txt', encoding="utf-8").readlines()]   # StopWords.txt需要自己准备
    # print(stopwords)
    pynlpir.open()
    sentences = pynlpir.segment(txt2, pos_tagging=False)
    #sentences = jieba.lcut(txt2)   # jieba的分词效果并不好, 这里我们使用pynlpir分词, 初始准确率非常高
    final_sentences = ''
    for sentence in sentences:
        if sentence not in stopwords:
            final_sentences += sentence
        else:
            final_sentences += ''
    with open(des, 'w', encoding="UTF-8") as f:
        f.write(txt2)
        
Clear_StopWords("E:/pycharm_project/data/train-2.txt", "E:/pycharm_project/data/train-3.txt")
  
  
  # lowercase_txt 把文本数据全部转换为小写, 统一格式, 用于防止后面dict的key出现因为大小写问题无法匹配
  def lowercase_txt(file_name):
    """
    file_name is the full path to the file to be opened
    """
    with open(file_name, 'r+', encoding = "utf8") as f:
        contents = f.read()  # read contents of file
        contents = contents.lower()  # convert to lower case
        f.seek(0, 0)  # position back to start of file
        f.write(contents)
        f.truncate()
        
        
 lowercase_txt('E:/pycharm_project/data/train-3.txt')
