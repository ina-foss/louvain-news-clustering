docker run -d --name news_tweets_0 -v /rex/local/bmtweet/news_twitter/:/usr/src/app docker-ina.rech.ina.fr/bmazoyer/igraph python /usr/src/app/main_work.py --path /usr/src/app/ --dataset event2018_short_0
docker run -d --name news_tweets_1 -v /rex/local/bmtweet/news_twitter/:/usr/src/app docker-ina.rech.ina.fr/bmazoyer/igraph python /usr/src/app/main_work.py --path /usr/src/app/ --dataset event2018_short_1
docker run -d --name news_tweets_2 -v /rex/local/bmtweet/news_twitter/:/usr/src/app docker-ina.rech.ina.fr/bmazoyer/igraph python /usr/src/app/main_work.py --path /usr/src/app/ --dataset event2018_short_2
docker run -d --name news_tweets_3 -v /rex/local/bmtweet/news_twitter/:/usr/src/app docker-ina.rech.ina.fr/bmazoyer/igraph python /usr/src/app/main_work.py --path /usr/src/app/ --dataset event2018_short_3
