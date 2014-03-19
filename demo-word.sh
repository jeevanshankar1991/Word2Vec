#!/bin/sh
## scipt for computing the vectors  and checking the accuracy with google's questions-words analogy task
pwd=$PWD
home=$HOME
factorielib="${home}/.m2/repository/cc/factorie/factorie/1.0-SNAPSHOT/factorie-1.0-SNAPSHOT.jar"
scalalib="${home}/.m2/repository/org/scala-lang/scala-library/2.10.2/scala-library-2.10.2.jar"
wordvec="cn.jar"
gcc distance.c -o distance -lm ## google's word2vec code
mvn compile; mvn compile;
if [ $? -eq 0 ]; then
 cd target/classes
 jar cf ${wordvec} .
 mv ${wordvec} ../../${wordvec}
 cd ../..
 wordvecjar=$pwd/$wordvec
 java -Xmx10g -cp "${wordvecjar}:${factorielib}:${scalalib}" WordVec --cbow=1 --train text8 --output vectors.txt --size=200 --window=5 --min-count=5 --threads=12 --negative=1 --save-vocab=text8.vocab
  if [ $? -eq 0 ]; then
    ./distance vectors.txt
  fi
fi

