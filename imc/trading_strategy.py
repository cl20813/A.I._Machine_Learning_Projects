

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data)
    


import math

def cdf(x: float) -> float:
    """
    Cumulative distribution function for the standard normal distribution.
    A simple approximation using the error function (erf).
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def black_scholes(
    asset_price: float,
    strike_price: float,
    expiration_time: float,
    risk_free_rate: float,
    volatility: float,
) -> float:
    """
    Calculates the price of a European call option using the Black-Scholes formula.
    """
    d1 = (math.log(asset_price / strike_price) + (risk_free_rate + volatility ** 2 / 2) * expiration_time) / (volatility * math.sqrt(expiration_time))
    d2 = d1 - volatility * math.sqrt(expiration_time)
    return asset_price * cdf(d1) - strike_price * math.exp(-risk_free_rate * expiration_time) * cdf(d2)


def implied_volatility(
    option_price: float,
    asset_price: float,
    strike_price: float,
    expiration_time: float,
    risk_free_rate: float,
) -> float:
    """
    Calculates the implied volatility of a European call option using the
    Black-Scholes model and a numerical bisection method.

    :param option_price: The observed market price of the option.
    :param asset_price: The current price of the underlying asset.
    :param strike_price: The option's strike price.
    :param expiration_time: The time to expiration in years.
    :param risk_free_rate: The risk-free interest rate.
    :return: The implied volatility as a decimal.
    """
    # Define a target function to find the root for.
    # We want to find the volatility where black_scholes(vol) - option_price = 0.
    def price_diff(vol: float) -> float:
        return black_scholes(
            asset_price,
            strike_price,
            expiration_time,
            risk_free_rate,
            vol
        ) - option_price

    # Set initial bounds for the bisection method.
    # Volatility is typically between 0.01% and 1000%
    low_vol = 0.0001
    high_vol = 10.0
    
    # Set the tolerance for the solution.
    epsilon = 1e-6
    
    # Perform the bisection search.
    for _ in range(100):  # Limit iterations to prevent infinite loops
        mid_vol = (low_vol + high_vol) / 2
        
        # If the difference is close enough, we have found our solution.
        if abs(price_diff(mid_vol)) < epsilon:
            return mid_vol
            
        # If the price is too low, the volatility guess is too low.
        if price_diff(low_vol) * price_diff(mid_vol) < 0:
            high_vol = mid_vol
        else:
            low_vol = mid_vol
            
    # If a solution wasn't found within the iterations, return the last guess.
    return (low_vol + high_vol) / 2