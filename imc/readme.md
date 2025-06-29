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

In round 3, ```VOLCANIC_ROCK_VOUCHER``` are call options at different strike prices. So we use Black-Scholes model for call option, note that   

```C = S.N(d1) - K* exp(-r*T)*N(d2)```   
```P = K * exp(-r * T) * N(d2) - S * N(-d1)```,   

where ```d2 = d1 - $\sigma \sqrt{T},$``` and ```d1 = $\frac{ \log(S/K) + (r+ \sigma^2/2)*T }{ \sigma * \sqrt{T} },$```   
```S = spot asset price, K = Strike price, r is interest```.    

## Reference
Black Scholes model is to compute individual price of a call or put option, but by using put-call parity formula, we can compute the put option price as well assuming there is no arbitrage pricing (guarantee a profit witout risk). But for reference, Black-Scholes model for put option is
, and
  
Put-call parity:     
C + PV(K) = P + S     
where PV(K) is the present value of the strike price K.   
  
For example, if C=3, PV=100, P=7, S=98, then    
$5+100 \neq 7+98$, this means either call option price is too low or put option price is too high. If we started by setting C from Black-scholes model and assume it is correctly modeled, then put option price is the one too high. Then we can exploit the price by belows:   
    
Sell the overprice put 7   
Buy the call option at 3   
Short stocks at 98 (borrow shares of a stock and sell immediately with intention of buying them back at lower price later and return to lender.)   
Invest in a bond (PV of strike) -$100 (from 98, invest it in a risk-free bond that will grow to exactly at the option's expiration date.)  
then you will have +7 -3 +98 - 100 = 2.   
  
Two cases:      
Stock price > K      
Call is exercised: you buy stock at K   (You have K from the the expired Bond)   
Deliver stock to cover your short       (Return the stock to the lender)   
Put expires worthless        

Stock price < K       
Put is exercised: you buy stock at K    (You sold the put option, and the buyer will exercise the put, and you are obligated to buy the stock at k)       
Use bond proceeds to pay K              (You have K from the the expired Bond)    
Deliver stock to cover your short       (Return the stock to the lender)   
Call expires worthless      

In both scenarios, you have $2 at the end.       





