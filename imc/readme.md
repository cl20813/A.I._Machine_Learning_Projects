## Set up

[jmerle][https://github.com/jmerle/imc-prosperity-3-backtester]

Go to terminal and type belows:

```git clone https://github.com/jmerle/imc-prosperity-3-backtester.git```
(it clones the repository into a new folder named imc-prosperity-3-backtester in your current working directory.)

```cd imc-prosperity-3-backtester```

```pip install -e .```

For reference,    
```which prosperity3bt```   returns ```/opt/anaconda3/envs/faiss_env/bin/prosperity3bt(root of run file)```    
But the package folder is located at  ```/Users/joonwonlee/imc-prosperity-3-backtester```. (cmd+shit+c and cmd+shit+.(period) to reveal folders in mac)   
Also prosperity3bt-0.0.0.dist-info is a meta data that helps upgrade or reinstallation.   

### Run example
```prosperity3bt /Users/joonwonlee/Documents/imc_trading/round3/test0629.py 3```       #round 3 all days
```prosperity3bt /Users/joonwonlee/Documents/imc_trading/round3/test0629.py 3-0```     #round 3 day 1


# Implied Volatility
Grid search for the range between (0.0001, 5.0)
Compute Black-Scholes price using the value above.
Compare Black-Scholes model price to the market price (option price) until two prices are close enough.

In round 3, ```VOLCANIC_ROCK_VOUCHER``` are call options at different strike prices. So we use Black-Scholes model for call option,

C = S.N(d1) - K* exp(-r*T)*N(d2), where
d2 = d1 - $\sigma \sqrt{T}$  
d1 = $\frac{\log{S/K} + (r+ \sigma^2/2)*T}{\sigma* \sqrt{T}}$


