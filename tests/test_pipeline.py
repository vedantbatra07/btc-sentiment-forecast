import json

import numpy as np
import pandas as pd
import pytest

from btc_sentiment.demo import populate
from btc_sentiment.features import PRICE_FEATURES, SENTIMENT_FEATURES, build_dataset, daily_sentiment, daily_prices
from btc_sentiment.ingest import normalize_post, fetch_posts, fetch_posts_parallel, fetch_news
from btc_sentiment.metrics import risk_metrics, backtest
from btc_sentiment.model import eligible_training, walk_forward, forecast_latest
from btc_sentiment.storage import connect, read, upsert
from btc_sentiment.cli import evaluate


@pytest.fixture
def database(tmp_path):
    con = connect(tmp_path/"test.sqlite")
    populate(con)
    yield con
    con.close()


def test_future_text_and_prices_cannot_change_earlier_features(database):
    before, _ = build_dataset(database)
    cutoff = before.index[95]
    database.execute("UPDATE posts SET text='terrible terrible terrible' WHERE timestamp>=?",(cutoff.isoformat(),))
    database.execute("UPDATE prices SET close=close*2, high=high*2 WHERE timestamp>=?",(cutoff.isoformat(),))
    after, _ = build_dataset(database)
    pd.testing.assert_frame_equal(before.loc[:cutoff,PRICE_FEATURES+SENTIMENT_FEATURES],after.loc[:cutoff,PRICE_FEATURES+SENTIMENT_FEATURES])


def test_targets_have_exact_daily_horizon(database):
    x, joined = build_dataset(database)
    day = x.index[40]
    for h in range(6):
        expected = np.log(joined.loc[day+pd.Timedelta(days=h),'close']/joined.loc[day-pd.Timedelta(days=1),'close'])
        assert x.loc[day,f'target_{h}'] == pytest.approx(expected)


def test_training_label_boundaries_and_no_duplicate_oos(database):
    x,_ = build_dataset(database)
    oos,audit = walk_forward(x)
    assert (pd.to_datetime(audit.latest_label_end) <= pd.to_datetime(audit.cutoff)).all()
    assert not oos.duplicated(['origin','horizon','model']).any()
    assert oos.groupby(['model','horizon']).size().nunique() == 1
    assert set(oos.horizon) == set(range(6))
    for h in range(6):
        cutoff = x.index[95]
        train=eligible_training(x,PRICE_FEATURES+SENTIMENT_FEATURES,h,cutoff)
        assert (train.index+pd.Timedelta(days=h+1) <= cutoff).all()


def test_test_labels_do_not_change_predictions_for_same_fold(database):
    x,_=build_dataset(database)
    oos,_=walk_forward(x)
    cutoff=oos.origin.min()
    changed=x.copy()
    for h in range(6):
        changed.loc[changed.index>=cutoff,f'target_{h}'] += .5
    altered,_=walk_forward(changed)
    end=cutoff+pd.Timedelta(days=14)
    np.testing.assert_allclose(oos.loc[oos.origin<end,'predicted_return'],altered.loc[altered.origin<end,'predicted_return'])


def test_scraping_gap_is_not_neutral_sentiment(database):
    coverage=read(database,'coverage')
    posts=read(database,'posts')
    idx=pd.date_range('2024-01-01',periods=3,tz='UTC')
    coverage=coverage[coverage.day!=idx[1].isoformat()]
    posts=posts[~posts.timestamp.str.startswith('2024-01-03')]
    s=daily_sentiment(posts,coverage,idx)
    assert np.isnan(s.loc[idx[1],'sentiment'])
    assert s.loc[idx[2],'sentiment']==0
    assert s.loc[idx[2],'post_count']==0


def test_missing_hour_invalidates_daily_ohlc(database):
    prices=read(database,'prices').iloc[1:]
    d=daily_prices(prices)
    assert pd.isna(d.iloc[0].close)
    assert d.iloc[0].hours==23


def test_engagement_cannot_change_model_features(database):
    before,_=build_dataset(database)
    database.execute('UPDATE posts SET likes=999999,reposts=999999,views=999999,followers=999999')
    after,_=build_dataset(database)
    pd.testing.assert_frame_equal(before,after)


def test_post_normalization_and_dst():
    row=normalize_post({'id':'1','text':'Bitcoin good','createdAt':'Sun Nov 03 06:30:00 +0000 2024',
        'likeCount':3,'viewCount':None,'author':{'followers':42}},'2024-11-04T00:00:00+00:00')
    assert row['followers']==42 and row['views'] is None
    assert pd.Timestamp(row['timestamp']).tz_convert('America/New_York').isoformat().endswith('-05:00')
    with pytest.raises(ValueError):
        normalize_post({'error':'invalid search'},'2024-01-01')


def test_drawdown_includes_initial_capital_and_zero_denominators():
    result=risk_metrics([-.1,.05])
    assert result['max_drawdown']==pytest.approx(-.1)
    assert result['total_return']==pytest.approx(-.055)
    assert risk_metrics([0,0])['sharpe'] is None
    assert risk_metrics([.01,.02])['sortino'] is None
    expected=np.sqrt(365)*np.mean([-.1,.05])/np.sqrt((.1**2)/2)
    assert result['sortino']==pytest.approx(expected)


def test_trading_executes_next_hour_not_previous_close():
    dates=pd.date_range('2024-01-01',periods=4,tz='UTC')
    joined=pd.DataFrame({'execution_open':[100,110,99,99]},index=dates)
    oos=pd.DataFrame({'origin':dates[:3],'horizon':0,'model':'price_only','predicted_return':[.1,.1,0]})
    metrics,curves=backtest(oos,joined,10)
    net=curves[curves.model=='price_only'].net_return
    np.testing.assert_allclose(net,[.099,-.1,-.001])
    assert len(metrics)==4


def test_live_forecast_refuses_missing_latest_sentiment(database):
    x,_=build_dataset(database)
    oos,_=walk_forward(x)
    x.loc[x.index[-1],'sentiment_1']=np.nan
    with pytest.raises(ValueError,match='incomplete'):
        forecast_latest(x,oos)


def test_synthetic_cannot_be_reported_as_live(tmp_path):
    db=tmp_path/'fixture.sqlite'
    with connect(db) as con:
        populate(con)
    with pytest.raises(ValueError,match='Synthetic'):
        evaluate(db,tmp_path/'report')


class Response:
    ok=True
    def __init__(self,data): self.data=data
    def json(self): return self.data


class ApifyHTTP:
    def __init__(self): self.starts=0
    def post(self,url,**kwargs):
        self.starts+=1
        assert kwargs['params']['maxTotalChargeUsd']==1
        assert 'since:2024-01-01 until:2024-01-02' in kwargs['json']['searchTerms'][0]
        return Response({'data':{'id':'run-1'}})
    def get(self,url,**kwargs):
        if 'actor-runs' in url:
            return Response({'data':{'status':'SUCCEEDED','defaultDatasetId':'dataset-1'}})
        return Response([{'id':'123','text':'Bitcoin is good','createdAt':'Mon Jan 01 12:00:00 +0000 2024','author':{}}])


def test_apify_adapter_and_completed_run_resume(tmp_path,monkeypatch):
    monkeypatch.setenv('APIFY_TOKEN','test-only')
    http=ApifyHTTP()
    with connect(tmp_path/'test.sqlite') as con:
        for _ in range(2):
            fetch_posts(con,pd.Timestamp('2024-01-01',tz='UTC'),pd.Timestamp('2024-01-02',tz='UTC'),1,http=http)
        assert http.starts==1
        assert len(read(con,'posts'))==1
        assert read(con,'coverage').status.iloc[0]=='SUCCEEDED'


def test_tavily_adapter_stores_publication_and_retrieval_times(tmp_path,monkeypatch):
    monkeypatch.setenv('TAVILY_API_KEY','test-only')
    class HTTP:
        def post(self,url,**kwargs):
            assert kwargs['json']['topic']=='news'
            return Response({'results':[{'title':'Bitcoin update','url':'https://example.org/news',
                'content':'Market update','published_date':'2024-01-01','score':.9}]})
    with connect(tmp_path/'test.sqlite') as con:
        assert fetch_news(con,HTTP())==1
        row=read(con,'news').iloc[0]
        assert row.published_at.startswith('2024-01-01')
        assert row.retrieved_at!=row.published_at


def test_parallel_scraping_allocates_fixed_total_budget(tmp_path,monkeypatch):
    import threading
    calls=[]
    lock=threading.Lock()
    def fake(con,start,end,max_charge_usd,per_day,write_config):
        with lock:
            calls.append((start,max_charge_usd))
        return 1
    monkeypatch.setattr('btc_sentiment.ingest.fetch_posts',fake)
    count=fetch_posts_parallel(tmp_path/'parallel.sqlite',pd.Timestamp('2024-01-01',tz='UTC'),
        pd.Timestamp('2024-01-05',tz='UTC'),.06,workers=4)
    assert count==4
    assert sum(c[1] for c in calls)==pytest.approx(.06)
    assert len(set(c[0] for c in calls))==4


def test_unknown_intermediate_target_is_excluded(database):
    x,_=build_dataset(database)
    day=x.index[110]
    x.loc[day,'target_2']=np.nan
    oos,_=walk_forward(x)
    assert day not in set(oos.origin)
