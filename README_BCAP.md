# Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy
This repository provides codes for [ICAIF 2020 paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996)
This ensemble strategy is reimplemented in a Jupiter Notebook at [FinRL](https://github.com/AI4Finance-LLC/FinRL-Library).
Hongyang Yang, Xiao-Yang Liu, Shan Zhong, and Anwar Walid. 2020. Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy. In ICAIF ’20: ACM International Conference on AI in Finance, Oct. 15–16, 2020, Manhattan, NY. ACM, New York, NY, USA.

# Project Folder Structure

```
.devcontainer/  
config/ 
    config.py : common configurations for the project
data/
    ^DJI.csv : DJI (index, not single stocks) data from 2.1.2009-19.08.2020, used for backtesting
    dow_30_2009_2020.csv : DJI data for each stock from  2.1.2009-19.08.2020, used for modeling and backtesting
    dow30_turbulence_index.csv : ?
    ETF_SPY_2009_2020.csv : ? 
env/
    EnvMultipleStock_train.py : training environment
    EnvMultipleStock_validation.py: validation environment
    EnvMultipleStock_trade.py: trade (~test) environment
figs/
    data.PNG : illustration of training/validation/trade process
    performance.png : displays cumulative return with transaction cost. Not automatically implemented in the code given.
    stock_trading.png : overview ofer DRL setup
model/
    modely.py : functions for all agents, DRL ensemble strategy etc.
preprocessing/
    preprocessors.py : data preprocessors, feature creators etc. Only run if set in run_.py.
results/
    firstRun : original results from paper
        account_rewards_trade_ensemble_126.csv : account trading rewards of ensemble strategy, after 126 iterations
        ...same up until iteration 1197, in steps of 63
        account_value_trade_ensemble_126.csv : account trading value of ensemble strategy, after 126 iterations
        account_value_trade_ensemble_126.png : corresponding plot
        ...same up until iteration 1197, in steps of 63
        account_value_train.csv : account training value 
        account_value_train.png : account training value plot
        account_value_validation_126.csv : account value of validation period after 126 iterations
        account_value_validation_126.png : corresponding plot
        ...same up until iteration 1197, in steps of 63
        last_state_ensemble_62.csv : ?
    _date_ : subsequent runs by me
trained_models/
    firstRun : original trained models from the paper
        firstrun.txt : all logs during runtime of the algorithm
                => for some reason I don't get this file when running it on my own, why?
        A2C_30k_dow_126.zip : A2C model with 30'000 time steps and after the 126th iteration
            data : 
            parameter_list : 
            parameters : 
        ...same up until iteration 1197, in steps of 63
        DDPG_10k_dow_126.zip : DDPG model with 10'000 time steps and after the 126th iteration
            data : 
            parameter_list : 
            parameters : 
        ...same up until iteration 1197, in steps of 63
        PPO_100k_dow_126.zip : PPO model with 100'000 time steps and after the 126th iteration
            data : 
            parameter_list : 
            parameters : 
        ...same up until iteration 1197, in steps of 63
backtesting.ipynb : backtesting implemented based on results from previous run and with quantopian.
done_data.csv : preprocessed data. Used by defaults unless specified in the python run_.py script.
run_original.py : original code as implemented for the paper, run ensemble strategy
run_ppo.py : created by me. run only PPO




....
```



