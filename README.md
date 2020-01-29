<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">In this post, we will be building a <strong class="id iq">semantic documents search engine</strong> by using <a class="bu dh iw ix iy iz" href="http://qwone.com/~jason/20Newsgroups/" target="_blank" rel="noopener nofollow">20newsgroup open-source dataset</a>.</p>
<h1 class="hb hc cs ax aw ee ka mm kc mn mo mp mq mr ms mt hf" data-selectable-paragraph="">Prerequisites</h1>
<ul class="">
<li class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io mz na nb" data-selectable-paragraph=""><a class="bu dh iw ix iy iz" href="https://www.python.org/" target="_blank" rel="noopener nofollow">Python 3.5</a>+</li>
<li class="ia ib cs ax id b ie nc ig nd mg ne mi nf mk ng io mz na nb" data-selectable-paragraph=""><a class="bu dh iw ix iy iz" href="https://pypi.org/project/pip/" target="_blank" rel="noopener nofollow">pip 19</a>+ or pip3</li>
<li class="ia ib cs ax id b ie nc ig nd mg ne mi nf mk ng io mz na nb" data-selectable-paragraph=""><a class="bu dh iw ix iy iz" href="https://www.nltk.org/" target="_blank" rel="noopener nofollow">NLTK</a></li>
<li class="ia ib cs ax id b ie nc ig nd mg ne mi nf mk ng io mz na nb" data-selectable-paragraph=""><a class="bu dh iw ix iy iz" href="https://scikit-learn.org/stable/" target="_blank" rel="noopener nofollow">Scikit-learn</a></li>
<li class="ia ib cs ax id b ie nc ig nd mg ne mi nf mk ng io mz na nb" data-selectable-paragraph=""><a class="bu dh iw ix iy iz" href="https://www.tensorflow.org" target="_blank" rel="noopener nofollow">TensorFlow-GPU</a></li>
</ul>
<h1 class="hb hc cs ax aw ee ka mm kc mn mo mp mq mr ms mt hf" data-selectable-paragraph="">1. Getting Ready</h1>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">For this post we will need the above prerequisites<strong class="id iq">,&nbsp;</strong>If you do not have it yet, please make ready for it.</p>
<h1 class="hb hc cs ax aw ee ka mm kc mn mo mp mq mr ms mt hf" data-selectable-paragraph="">2. Data collection</h1>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">Here, we are using 20newsgroup dataset to the analysis of a text search engine giving input keywords/sentences input.</p>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">The 20 Newsgroups data set is a collection of approximately 11K newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups.</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">news = pd.read_json('<a class="bu dh iw ix iy iz" href="https://raw.githubusercontent.com/zayedrais/DocumentSearchEngine/master/data/newsgroups.json" target="_blank" rel="noopener nofollow">https://raw.githubusercontent.com/zayedrais/DocumentSearchEngine/master/data/newsgroups.json</a>')</span></pre>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">2.1 data cleaning:</h2>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">Before going into a clean phase, we are retrieving the subject of the document from the text.</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">for i,txt in enumerate(news['content']):<br />    subject = re.findall('Subject:(.*\n)',txt)<br />    if (len(subject) !=0):<br />        news.loc[i,'Subject'] =str(i)+' '+subject[0]<br />    else:<br />        news.loc[i,'Subject'] ='NA'<br />df_news =news[['Subject','content']]</span></pre>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">Now, we are removing the unwanted data from text content and the subject of a dataset.</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">df_news.content =df_news.content.replace(to_replace='from:(.*\n)',value='',regex=True) ##remove from to email <br />df_news.content =df_news.content.replace(to_replace='lines:(.*\n)',value='',regex=True)<br />df_news.content =df_news.content.replace(to_replace='[!"#$%&amp;\'()*+,/:;&lt;=&gt;?@[\\]^_`{|}~]',value=' ',regex=True) #remove punctuation except<br />df_news.content =df_news.content.replace(to_replace='-',value=' ',regex=True)<br />df_news.content =df_news.content.replace(to_replace='\s+',value=' ',regex=True)    #remove new line<br />df_news.content =df_news.content.replace(to_replace='  ',value='',regex=True)                #remove double white space<br />df_news.content =df_news.content.apply(lambda x:x.strip())  # Ltrim and Rtrim of whitespace</span></pre>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">2.2 data preprocessing</h2>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">Preprocessing is one of the major steps when we are dealing with any kind of text models. During this stage, we have to look at the distribution of our data, what techniques are needed and how deep we should clean.</p>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Lowercase</h2>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">Conversion the text into a lower form. i.e. &lsquo;<strong class="id iq">Dogs&rsquo;</strong> into &lsquo;<strong class="id iq">dogs</strong>&rsquo;</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">df_news['content']=[entry.lower() for entry in df_news['content']]</span></pre>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Word Tokenization</h2>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">Word tokenization is the process to divide the sentence into the form of a word.</p>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">&ldquo;<strong class="id iq">Jhon is running in the track</strong>&rdquo; &rarr; &lsquo;<strong class="id iq">john</strong>&rsquo;, &lsquo;<strong class="id iq">is</strong>&rsquo;, &lsquo;<strong class="id iq">running</strong>&rsquo;, &lsquo;<strong class="id iq">in</strong>&rsquo;, &lsquo;<strong class="id iq">the</strong>&rsquo;, &lsquo;<strong class="id iq">track</strong>&rsquo;</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">df_news['Word tokenize']= [word_tokenize(entry) for entry in df_news.content]</span></pre>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Stop words</h2>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">Stop words are the most commonly occurring words which don&rsquo;t give any additional value to the document vector. in-fact removing these will increase computation and space efficiency. <a class="bu dh iw ix iy iz" href="https://www.nltk.org/" target="_blank" rel="noopener nofollow">NLTK</a> library has a method to download the stopwords.</p>
<figure class="lp lq lr ls lt gp t u paragraph-image">
<div class="t u nz">
<div class="gu y br gv">
<div class="oa y"><img class="is it z ab ac fu v ha fr-fic fr-dii fr-draggable" src="https://miro.medium.com/max/601/1*PdgWsOM1ep9Z2rfkQ6UJZA.png" width="601" height="275" data-fr-image-pasted="true" /></div>
</div>
</div>
</figure>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Word Lemmatization</h2>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">Lemmatisation is a way to reduce the word to root synonym of a word. Unlike Stemming, Lemmatisation makes sure that the reduced word is again a dictionary word (word present in the same language). WordNetLemmatizer can be used to lemmatize any word.</p>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">i.e. <strong class="id iq">rocks &rarr;rock, better &rarr;good, corpora &rarr;corpus</strong></p>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">Here created wordLemmatizer function to remove a <strong class="id iq">single character</strong>, <strong class="id iq">stopwords</strong> and <strong class="id iq">lemmatize</strong> the words.</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph=""># WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun<br />def wordLemmatizer(data):<br />    tag_map = defaultdict(lambda : wn.NOUN)<br />    tag_map['J'] = wn.ADJ<br />    tag_map['V'] = wn.VERB<br />    tag_map['R'] = wn.ADV<br />    file_clean_k =pd.DataFrame()<br />    for index,entry in enumerate(data):<br />        <br />        # Declaring Empty List to store the words that follow the rules for this step<br />        Final_words = []<br />        # Initializing WordNetLemmatizer()<br />        word_Lemmatized = WordNetLemmatizer()<br />        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.<br />        for word, tag in pos_tag(entry):<br />            # Below condition is to check for Stop words and consider only alphabets<br />            if len(word)&gt;1 and word not in stopwords.words('english') and word.isalpha():<br />                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])<br />                Final_words.append(word_Final)<br />            # The final processed set of words for each iteration will be stored in 'text_final'<br />                file_clean_k.loc[index,'Keyword_final'] = str(Final_words)<br />                file_clean_k.loc[index,'Keyword_final'] = str(Final_words)<br />                file_clean_k=file_clean_k.replace(to_replace ="\[.", value = '', regex = True)<br />                file_clean_k=file_clean_k.replace(to_replace ="'", value = '', regex = True)<br />                file_clean_k=file_clean_k.replace(to_replace =" ", value = '', regex = True)<br />                file_clean_k=file_clean_k.replace(to_replace ='\]', value = '', regex = True)<br />    return file_clean_k</span></pre>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">By using this function took around <strong class="id iq">13 hrs</strong> time to check and lemmatize the words of 11K documents of the 20newsgroup dataset. Find below the JSON file of the lemmatized word.</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph=""><a class="bu dh iw ix iy iz" href="https://raw.githubusercontent.com/zayedrais/DocumentSearchEngine/master/data/WordLemmatize20NewsGroup.json" target="_blank" rel="noopener nofollow">https://raw.githubusercontent.com/zayedrais/DocumentSearchEngine/master/data/WordLemmatize20NewsGroup.json</a></span></pre>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">2.3 data is ready for use</h2>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">See a sample of clean data-</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">df_news.Clean_Keyword[0]</span></pre>
<figure class="lp lq lr ls lt gp t u paragraph-image">
<div class="lw lx br ly v">
<div class="t u ob">
<div class="gu y br gv">
<div class="oc y"><img class="is it z ab ac fu v ha fr-fic fr-dii fr-draggable" src="https://miro.medium.com/max/718/1*Br5cASjTPcoN0J1QhXIYuA.png" width="718" height="105" data-fr-image-pasted="true" /></div>
</div>
</div>
</div>
</figure>
<h1 class="hb hc cs ax aw ee ka mm kc mn mo mp mq mr ms mt hf" data-selectable-paragraph="">3. Document Search engine</h1>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">In this post, we are using three approaches to understand text analysis.</p>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">1.Document search engine with <strong class="id iq">TF-IDF</strong></p>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">2.Document search engine with <strong class="id iq">Google Universal sentence encoder</strong></p>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">3.1 Calculating the ranking by using <a class="bu dh iw ix iy iz" href="https://en.wikipedia.org/wiki/Cosine_similarity" target="_blank" rel="noopener nofollow">cosine similarity</a></h2>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">It is the most common metric used to calculate the similarity between document text from input keywords/sentences. Mathematically, it measures the cosine of the angle b/w two vectors projected in a multi-dimensional space.</p>
<figure class="lp lq lr ls lt gp t u paragraph-image">
<div class="t u od">
<div class="gu y br gv">
<div class="oe y"><img class="is it z ab ac fu v ha fr-fic fr-dii fr-draggable" src="https://miro.medium.com/max/372/1*nJT7q9nlDWgXllSHcI4ZJA.jpeg" width="372" height="263" data-fr-image-pasted="true" /></div>
</div>
</div>
<figcaption class="bb bp ma dx mb w t u mc md aw fa" data-selectable-paragraph="">Cosine Similarity b/w document to query</figcaption>
</figure>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">In the above diagram, have 3 document vector value and one query vector in space. when we are calculating the cosine similarity b/w above 3 documents. The most similarity value will be D3 document from three documents.</p>
<h1 class="hb hc cs ax aw ee ka mm kc mn mo mp mq mr ms mt hf" data-selectable-paragraph="">1. Document search engine with TF-IDF:</h1>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph=""><a class="bu dh iw ix iy iz" href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf" target="_blank" rel="noopener nofollow"><strong class="id iq">TF-IDF</strong></a> stands for <strong class="id iq">&ldquo;Term Frequency &mdash; Inverse Document Frequency&rdquo;</strong>. This is a technique to calculate the weight of each word signifies the importance of the word in the document and corpus. This algorithm is mostly using for the retrieval of information and text mining field.</p>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Term Frequency (TF)</h2>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">The number of times a word appears in a document divided by the total number of words in the document. Every document has its term frequency.</p>
<figure class="lp lq lr ls lt gp t u paragraph-image">
<div class="lw lx br ly v">
<div class="t u of">
<div class="gu y br gv">
<div class="og y"><img class="is it z ab ac fu v ha fr-fic fr-dii fr-draggable" src="https://miro.medium.com/max/343/0*0Uzik-cTMA-i6BUt.png" width="343" height="121" data-fr-image-pasted="true" /></div>
</div>
</div>
</div>
</figure>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Inverse Data Frequency (IDF)</h2>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">The log of the number of documents divided by the number of documents that contain the word <strong class="id iq"><em class="ic">w</em></strong>. Inverse data frequency determines the weight of rare words across all documents in the corpus.</p>
<figure class="lp lq lr ls lt gp t u paragraph-image">
<div class="t u oh">
<div class="gu y br gv">
<div class="oi y"><img class="is it z ab ac fu v ha fr-fic fr-dii fr-draggable" src="https://miro.medium.com/max/390/0*t2Uxb_43L3vjwDPm.png" width="390" height="123" data-fr-image-pasted="true" /></div>
</div>
</div>
</figure>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">Lastly, the <strong class="id iq">TF-IDF</strong> is simply the TF multiplied by IDF.</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph=""><strong class="nk iq">TF-IDF = Term Frequency (TF) * Inverse Document Frequency (IDF)</strong></span></pre>
<figure class="lp lq lr ls lt gp t u paragraph-image">
<div class="t u oj">
<div class="gu y br gv">
<div class="ok y"><img class="is it z ab ac fu v ha fr-fic fr-dii fr-draggable" src="https://miro.medium.com/max/505/0*yJm1bH6Ds0vFFyhP.png" width="505" height="128" data-fr-image-pasted="true" /></div>
</div>
</div>
</figure>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">Rather than manually implementing <a class="bu dh iw ix iy iz" href="http://www.tfidf.com/" target="_blank" rel="noopener nofollow">TF-IDF</a> ourselves, we could use the class provided by <a class="bu dh iw ix iy iz" href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html" target="_blank" rel="noopener nofollow">Sklearn</a>.</p>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Generated TF-IDF by using TfidfVectorizer from Sklearn</h2>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">Import the packages:</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">import pandas as pd<br />import numpy as np<br />import os <br />import re<br />import operator<br />import nltk <br />from nltk.tokenize import word_tokenize<br />from nltk import pos_tag<br />from nltk.corpus import stopwords<br />from nltk.stem import WordNetLemmatizer<br />from collections import defaultdict<br />from nltk.corpus import wordnet as wn<br />from sklearn.feature_extraction.text import TfidfVectorizer</span></pre>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">TF-IDF</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">from sklearn.feature_extraction.text import TfidfVectorizer<br />import operator</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph="">## Create Vocabulary<br />vocabulary = set()</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph="">for doc in df_news.Clean_Keyword:<br />    vocabulary.update(doc.split(','))</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph="">vocabulary = list(vocabulary)</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph=""># Intializating the tfIdf model<br />tfidf = TfidfVectorizer(vocabulary=vocabulary)</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph=""># Fit the TfIdf model<br />tfidf.fit(df_news.Clean_Keyword)</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph=""># Transform the TfIdf model<br />tfidf_tran=tfidf.transform(df_news.Clean_Keyword)</span></pre>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">The above code has created TF-IDF weight of the whole dataset, Now have to create a function to generate a vector for the input query.</p>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Create a vector for Query/search keywords</h2>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">def gen_vector_T(tokens):</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph="">Q = np.zeros((len(vocabulary)))    <br />    x= tfidf.transform(tokens)<br />    #print(tokens[0].split(','))<br />    for token in tokens[0].split(','):<br />        #print(token)<br />        try:<br />            ind = vocabulary.index(token)<br />            Q[ind]  = x[0, tfidf.vocabulary_[token]]<br />        except:<br />            pass<br />    return Q</span></pre>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Cosine Similarity function for the calculation</h2>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">def cosine_sim(a, b):<br />    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))<br />    return cos_sim</span></pre>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Cosine Similarity b/w document to query function</h2>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">def cosine_similarity_T(k, query):<br />    preprocessed_query = preprocessed_query = re.sub("\W+", " ", query).strip()<br />    tokens = word_tokenize(str(preprocessed_query))<br />    q_df = pd.DataFrame(columns=['q_clean'])<br />    q_df.loc[0,'q_clean'] =tokens<br />    q_df['q_clean'] =wordLemmatizer(q_df.q_clean)<br />    d_cosines = []<br />    <br />    query_vector = gen_vector_T(q_df['q_clean'])<br />    for d in tfidf_tran.A:<br />        d_cosines.append(cosine_sim(query_vector, d))<br />                    <br />    out = np.array(d_cosines).argsort()[-k:][::-1]<br />    #print("")<br />    d_cosines.sort()<br />    a = pd.DataFrame()<br />    for i,index in enumerate(out):<br />        a.loc[i,'index'] = str(index)<br />        a.loc[i,'Subject'] = df_news['Subject'][index]<br />    for j,simScore in enumerate(d_cosines[-k:][::-1]):<br />        a.loc[j,'Score'] = simScore<br />    return a</span></pre>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Testing the function</h2>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">cosine_similarity_T(10,&rsquo;computer science&rsquo;)</span></pre>
<figure class="lp lq lr ls lt gp t u paragraph-image">
<div class="t u oq">
<div class="gu y br gv">
<div class="or y"><img class="is it z ab ac fu v ha fr-fic fr-dii fr-draggable" src="https://miro.medium.com/max/429/1*iSxlgzrxMz9Epnp4WkhxBQ.png" width="429" height="210" data-fr-image-pasted="true" /></div>
</div>
</div>
<figcaption class="bb bp ma dx mb w t u mc md aw fa" data-selectable-paragraph=""><strong class="aw iq">Result of top 5 similarity documents for &ldquo;computer science&rdquo; word</strong></figcaption>
</figure>
<h1 class="hb hc cs ax aw ee ka mm kc mn mo mp mq mr ms mt hf" data-selectable-paragraph="">2. Document search engine with Google Universal sentence encoder</h1>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Introduction Google USE</h2>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">The pre-trained <a class="bu dh iw ix iy iz" href="https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html" target="_blank" rel="noopener nofollow">Universal Sentence Encoder</a> is publicly available in <a class="bu dh iw ix iy iz" href="https://www.tensorflow.org/hub/" target="_blank" rel="noopener nofollow">Tensorflow-hub</a>. It comes with two variations i.e. one trained with <a class="bu dh iw ix iy iz" href="https://tfhub.dev/google/universal-sentence-encoder-large/5" target="_blank" rel="noopener nofollow"><strong class="id iq">Transformer encoder</strong></a> and others trained with <a class="bu dh iw ix iy iz" href="https://tfhub.dev/google/universal-sentence-encoder/4" target="_blank" rel="noopener nofollow"><strong class="id iq">Deep Averaging Network (DAN)</strong></a>. They are pre-trained on a large corpus and can be used in a variety of tasks (sentimental analysis, classification and so on). The two have a trade-off of accuracy and computational resource requirement. While the one with Transformer encoder has higher accuracy, it is computationally more expensive. The one with DNA encoding is computationally less expensive and with little lower accuracy.</p>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">Here we are using Second one DAN Universal sentence encoder as available in this URL:- <a class="bu dh iw ix iy iz" href="https://tfhub.dev/google/universal-sentence-encoder/4" target="_blank" rel="noopener nofollow">Google USE DAN Model</a></p>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">Both models take a word, sentence or a paragraph as input and output a <strong class="id iq">512</strong>-dimensional vector.</p>
<figure class="lp lq lr ls lt gp t u paragraph-image">
<div class="t u hw">
<div class="gu y br gv">
<div class="os y"><img class="is it z ab ac fu v ha fr-fic fr-dii fr-draggable" src="https://miro.medium.com/max/640/0*pGl0O-Z_Way5sT_U" width="640" height="189" data-fr-image-pasted="true" /></div>
</div>
</div>
<figcaption class="bb bp ma dx mb w t u mc md aw fa" data-selectable-paragraph="">A prototypical semantic retrieval pipeline, used for textual similarity.</figcaption>
</figure>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">Before using the TensorFlow-hub model.</p>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph=""><strong class="id iq">Prerequisite :</strong></p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">!pip install --upgrade tensorflow-gpu<br /> #Install TF-Hub.<br />!pip install tensorflow-hub<br />!pip install seaborn</span></pre>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">Now import the packages:</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">import pandas as pd<br />import numpy as np<br />import re, string<br />import os <br />import tensorflow as tf<br />import tensorflow_hub as hub<br />import matplotlib.pyplot as plt<br />import seaborn as sns<br />from sklearn.metrics.pairwise import linear_kernel</span></pre>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">Download the model from <a class="bu dh iw ix iy iz" href="https://tfhub.dev/google/universal-sentence-encoder/4" target="_blank" rel="noopener nofollow">TensorFlow-hub</a> of calling direct URL:</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">! curl -L -o 4.tar.gz "<a class="bu dh iw ix iy iz" href="https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed" target="_blank" rel="noopener nofollow">https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed</a>"</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph="">or<br />module_url = "<a class="bu dh iw ix iy iz" href="https://tfhub.dev/google/universal-sentence-encoder/4" target="_blank" rel="noopener nofollow">https://tfhub.dev/google/universal-sentence-encoder/4</a>"</span></pre>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph="">Load the Google DAN Universal sentence encoder</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">#Model load through local path:</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph="">module_path ="/home/zettadevs/GoogleUSEModel/USE_4"<br />%time model = hub.load(module_path)</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph="">#Create function for using model training<br />def embed(input):<br />    return model(input)</span></pre>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Use Case 1:- Word semantic</h2>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">WordMessage =[&lsquo;big data&rsquo;, &lsquo;millions of data&rsquo;, &lsquo;millions of records&rsquo;,&rsquo;cloud computing&rsquo;,&rsquo;aws&rsquo;,&rsquo;azure&rsquo;,&rsquo;saas&rsquo;,&rsquo;bank&rsquo;,&rsquo;account&rsquo;]</span></pre>
<figure class="lp lq lr ls lt gp t u paragraph-image">
<div class="t u ot">
<div class="gu y br gv">
<div class="ou y"><img class="is it z ab ac fu v ha fr-fic fr-dii fr-draggable" src="https://miro.medium.com/max/472/1*DfKpd8bPS1PA-UugtEDCFA.png" width="472" height="380" data-fr-image-pasted="true" /></div>
</div>
</div>
</figure>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Use Case 2: Sentence Semantic</h2>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">SentMessage =['How old are you?','what is your age?','how are you?','how you doing?']</span></pre>
<figure class="lp lq lr ls lt gp t u paragraph-image">
<div class="t u ov">
<div class="gu y br gv">
<div class="ow y"><img class="is it z ab ac fu v ha fr-fic fr-dii fr-draggable" src="https://miro.medium.com/max/467/1*-eUVfSMSsv8rmNoqPTmQ2w.png" width="467" height="375" data-fr-image-pasted="true" /></div>
</div>
</div>
</figure>
<h2 class="nj hc cs ax aw ee no np nq nr ns nt nu nv nw nx ny" data-selectable-paragraph="">Use Case 3: Word, Sentence and paragram Semantic</h2>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">word ='Cloud computing'</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph="">Sentence = 'what is cloud computing'</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph="">Para =("Cloud computing is the latest generation technology with a high IT infrastructure that provides us a means by which we can use and utilize the applications as utilities via the internet."<br />        "Cloud computing makes IT infrastructure along with their services available 'on-need' basis." <br />        "The cloud technology includes - a development platform, hard disk, computing power, software application, and database.")</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph="">Para5 =(<br />    "Universal Sentence Encoder embeddings also support short paragraphs. "<br />    "There is no hard limit on how long the paragraph is. Roughly, the longer "<br />    "the more 'diluted' the embedding will be.")</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph="">Para6 =("Azure is a cloud computing platform which was launched by Microsoft in February 2010."<br />       "It is an open and flexible cloud platform which helps in development, data storage, service hosting, and service management."<br />       "The Azure tool hosts web applications over the internet with the help of Microsoft data centers.")<br />case4Message=[word,Sentence,Para,Para5,Para6]</span></pre>
<figure class="lp lq lr ls lt gp t u paragraph-image">
<div class="t u ox">
<div class="gu y br gv">
<div class="oy y"><img class="is it z ab ac fu v ha fr-fic fr-dii fr-draggable" src="https://miro.medium.com/max/556/1*ejcbdMkwG1nUHBMtYUyjTg.png" width="556" height="156" data-fr-image-pasted="true" /></div>
</div>
</div>
</figure>
<h1 class="hb hc cs ax aw ee ka mm kc mn mo mp mq mr ms mt hf" data-selectable-paragraph="">Training the model</h1>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">Here we have trained the dataset at batch-wise because it takes a long time to execution to generate the graph of the dataset. so better to train batch-wise data.</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">Model_USE= embed(df_news.content[0:2500])</span></pre>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph=""><strong class="id iq">Save the model</strong>, for reusing the model.</p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">exported = tf.train.Checkpoint(v=tf.Variable(Model_USE))<br />exported.f = tf.function(<br />    lambda  x: exported.v * x,<br />    input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])</span><span class="nj hc cs ax nk b bp ol om on oo op nm y nn" data-selectable-paragraph="">tf.saved_model.save(exported,'/home/zettadevs/GoogleUSEModel/TrainModel')</span></pre>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph=""><strong class="id iq">Load the model from path:</strong></p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">imported = tf.saved_model.load(&lsquo;/home/zettadevs/GoogleUSEModel/TrainModel/&rsquo;)<br />loadedmodel =imported.v.numpy()</span></pre>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph=""><strong class="id iq">Function for document search:</strong></p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">def SearchDocument(query):<br />    q =[query]<br />    # embed the query for calcluating the similarity<br />    Q_Train =embed(q)<br />    <br />    #imported_m = tf.saved_model.load('/home/zettadevs/GoogleUSEModel/TrainModel')<br />    #loadedmodel =imported_m.v.numpy()<br />    # Calculate the Similarity<br />    linear_similarities = linear_kernel(Q_Train, con_a).flatten() <br />    #Sort top 10 index with similarity score<br />    Top_index_doc = linear_similarities.argsort()[:-11:-1]<br />    # sort by similarity score<br />    linear_similarities.sort()<br />    a = pd.DataFrame()<br />    for i,index in enumerate(Top_index_doc):<br />        a.loc[i,'index'] = str(index)<br />        a.loc[i,'File_Name'] = df_news['Subject'][index] ## Read File name with index from File_data DF<br />    for j,simScore in enumerate(linear_similarities[:-11:-1]):<br />        a.loc[j,'Score'] = simScore<br />    return a</span></pre>
<article>
<section class="gj gk gl gm gn">
<div class="n p">
<div class="ai aj ak al am jw ao v">
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph=""><strong class="id iq">Test the search:</strong></p>
<pre class="lp lq lr ls lt nh ni eu"><span class="nj hc cs ax nk b bp nl nm y nn" data-selectable-paragraph="">SearchDocument('computer science')</span></pre>
<figure class="lp lq lr ls lt gp t u paragraph-image">
<div class="t u oz">
<div class="gu y br gv">
<div class="pa y"><img class="is it z ab ac fu v ha fr-fic fr-dii fr-draggable" src="https://miro.medium.com/max/430/1*8Btzc5dq-HlTzCaFLw_jdQ.png" width="430" height="211" data-fr-image-pasted="true" /></div>
</div>
</div>
</figure>
<h1 class="hb hc cs ax aw ee ka mm kc mn mo mp mq mr ms mt hf" data-selectable-paragraph="">Conclusion:</h1>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph="">At the end of this tutorial, we are concluding that &ldquo;google universal sentence encoder&rdquo; model is providing the semantic search result while TF-IDF model doesn&rsquo;t know the meaning of the word. just giving the result based on words available on the documents.</p>
<p class="ia ib cs ax id b ie mu ig mv mg mw mi mx mk my io gj" data-selectable-paragraph=""><a href="https://medium.com/@zayedrais/build-your-semantic-document-search-engine-with-tf-idf-and-google-use-c836bf5f27fb" target="_blank" rel="noopener">Original Post on medium</a></p>
<p class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io gj" data-selectable-paragraph=""><strong class="id iq">Some references:</strong></p>
<ul class="">
<li class="ia ib cs ax id b ie me ig mf mg mh mi mj mk ml io mz na nb" data-selectable-paragraph=""><a class="bu dh iw ix iy iz" href="https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089" target="_blank" rel="noopener">TF-IDF</a></li>
<li class="ia ib cs ax id b ie nc ig nd mg ne mi nf mk ng io mz na nb" data-selectable-paragraph=""><a class="bu dh iw ix iy iz" href="https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html" target="_blank" rel="noopener nofollow">Google USE</a></li>
</ul>
</div>
</div>
</section>
</article>
<div class="is gi pb ji v pi pg sf" data-test-id="post-sidebar">&nbsp;</div>
<div class="ej">&nbsp;</div>
