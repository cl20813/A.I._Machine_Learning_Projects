
## ⚙️ Setup

To get started, I will use the package from below. Open your terminal and run the following commands:

```bash
git clone https://github.com/jmerle/imc-prosperity-3-backtester.git
cd imc-prosperity-3-backtester
pip install -e .
```

## 🪨 Round 1 and 2: MARKET MAKING STRATEGY

This market-making strategy aims to profit from the bid-ask spread by placing orders around a derived true value. It prioritizes managing its current inventory by widening its spread when it holds a significant position. The strategy also includes a mechanism for aggressively reducing its position (soft or hard liquidation) if it repeatedly hits its trading limit, which is a key component for managing risk and avoiding being stuck with a large, unwanted inventory.
   
## 🪨 Round 3: Black-Scholes Model, and Implied Volatility.

In Round 3, `VOLCANIC_ROCK_VOUCHER` represents **call options** with various strike prices. We use the **Black-Scholes formula for call options**:

### Black-Scholes Formulas:

- **Call Option (C):**  
  $$C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$

- **Put Option (P):**  
  $$P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)$$

Where:

- $$d_1 = \frac{\log(S/K) + (r + \sigma^2 / 2) \cdot T}{\sigma \cdot \sqrt{T}}$$  
- $$d_2 = d_1 - \sigma \cdot \sqrt{T}$$

## 📈 Implied Volatility and the Black-Scholes Model

In real markets, we observe the **option price** (e.g., the price of a voucher), but we do **not** observe volatility directly. To estimate volatility, we **invert the Black-Scholes formula** to find the value of **σ (sigma)** that makes the model price match the market price. This estimated volatility is known as **implied volatility**.

---

## 🔍 Grid Search for Implied Volatility

We perform a **grid search** over a range of volatility values (e.g., from `0.0001` to `5.0`). For each candidate volatility:

1. Compute the **Black-Scholes price** using the current volatility.
2. Compare the model price to the **observed market price**.
3. Repeat until the model price is sufficiently close to the market price.

---

---

## 📘 Definitions:

- **S** = Spot price of the underlying asset  
- **K** = Strike price  
- **r** = Risk-free interest rate  
- **T** = Time to maturity (in years)  
- **σ** = Volatility (standard deviation of returns)  
- **N(·)** = Cumulative distribution function of the standard normal distribution

---

## 🧠 Intuition:

- **d₁** measures how far **in-the-money** the option is, adjusted for volatility and time.
- **d₂** represents the **risk-neutral probability** that the option will be exercised.



## 📚 Reference

The **Black-Scholes model** is used to compute the theoretical price of individual **call** or **put** options. However, using the **put-call parity** formula, we can derive the price of a put option from a call option (or vice versa), assuming **no arbitrage** (i.e., no risk-free profit opportunities).

### 🧮 Put-Call Parity:

\[
C + PV(K) = P + S
\]

Where:
- **C** = Call option price  
- **P** = Put option price  
- **S** = Spot price of the underlying asset  
- **PV(K)** = Present value of the strike price **K**, discounted at the risk-free rate

---

### 🧪 Example:

Suppose:
- Call price \( C = 3 \)
- Present value of strike \( PV(K) = 100 \)
- Put price \( P = 7 \)
- Spot price \( S = 98 \)

Then:

$$
3 + 100 \neq 7 + 98 \Rightarrow 103 \neq 105
$$

This indicates a mispricing. If we assume the **call price is correct** (based on the Black-Scholes model), then the **put is overpriced**. We can exploit this arbitrage opportunity as follows:

---

### 💼 Arbitrage Strategy:

1. **Sell** the overpriced put at \$7  
2. **Buy** the call at \$3  
3. **Short** the stock at \$98  
4. **Invest** \$100 in a risk-free bond (to receive \$K at expiration)

**Net cash flow today:**
\[
+7 - 3 + 98 - 100 = +2
\]

---

### 📊 Outcomes at Expiration:

#### Case 1: Stock price **> K**
- Call is exercised: Buy stock at **K** using bond proceeds
- Cover short position by delivering the stock
- Put expires worthless
- **Profit = \$2**

#### Case 2: Stock price **< K**
- Put is exercised: Buy stock at **K** (obligation from sold put)
- Use bond proceeds to pay **K**
- Cover short position by delivering the stock
- Call expires worthless
- **Profit = \$2**

---

In both scenarios, the arbitrage strategy yields a **risk-free profit of \$2**, demonstrating the power of **put-call parity** in identifying mispriced options.
  

### 📁 Additional Notes

To verify the installed path of the CLI tool, run:

```bash
which prosperity3bt
```

For reference,    
```which prosperity3bt```   returns ```/opt/anaconda3/envs/faiss_env/bin/prosperity3bt(root of run file)```    
The actual package source code may be located at: ```/Users/joonwonlee/imc-prosperity-3-backtester```. (cmd+shit+c and cmd+shit+.(period) to reveal folders in mac)   
  

### Run example
```prosperity3bt /Users/joonwonlee/Documents/imc_trading/round3/test0629.py 3```          # round 3 all days     
```prosperity3bt /Users/joonwonlee/Documents/imc_trading/round3/test0629.py 3-0```        # round 3 day 1   



