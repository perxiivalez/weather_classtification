import numpy as np


def softmax(Z):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def relu(Z):
    return np.maximum(0.0, Z)


class MLP:
    def __init__(self, sizes, seed=0, l2=1e-3, dropout=0.0):
        self.sizes, self.l2, self.dropout = list(sizes), l2, dropout
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / m), (m, n)) for m, n in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((1, n)) for n in sizes[1:]]
        self._adam = None

    
    def forward(self, X, train=False, rng=None):
        
        A, cache = X, [dict(A=X)]
        L = len(self.W)
        for l in range(L):
            Z = A @ self.W[l] + self.b[l]               
            if l < L - 1:
                A = relu(Z)
                mask = None
                if train and self.dropout > 0:             
                    mask = (rng.random(A.shape) >= self.dropout) / (1.0 - self.dropout)
                    A = A * mask
                cache.append(dict(Z=Z, A=A, mask=mask))
            else:
                P = softmax(Z)
                cache.append(dict(Z=Z, A=P))
        return P, cache

    def predict_proba(self, X, T=1.0):
        if T == 1.0:
            return self.forward(X)[0]
        return softmax(self.forward(X)[1][-1]["Z"] / T)

  
    def loss(self, P, y):
        n = len(y)
        ce = -np.log(P[np.arange(n), y] + 1e-12).mean()
        reg = 0.5 * self.l2 * sum((W ** 2).sum() for W in self.W)
        return ce + reg

 
    def backward(self, y, cache):
        
        n, L = len(y), len(self.W)
        P = cache[-1]["A"]
        dZ = P.copy()
        dZ[np.arange(n), y] -= 1.0                        
        dZ /= n
        gW, gb = [None] * L, [None] * L
        for l in range(L - 1, -1, -1):
            A_prev = cache[l]["A"]                        
            gW[l] = A_prev.T @ dZ + self.l2 * self.W[l]   
            gb[l] = dZ.sum(axis=0, keepdims=True)         
            if l > 0:
                dA = dZ @ self.W[l].T                     
                if cache[l]["mask"] is not None:
                    dA = dA * cache[l]["mask"]
                dZ = dA * (cache[l]["Z"] > 0)             
        return gW, gb

   
    def step(self, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        if self._adam is None:
            self._adam = dict(t=0, mW=[np.zeros_like(w) for w in self.W], vW=[np.zeros_like(w) for w in self.W],
                              mb=[np.zeros_like(b) for b in self.b], vb=[np.zeros_like(b) for b in self.b])
        a = self._adam
        a["t"] += 1
        c1, c2 = 1 - b1 ** a["t"], 1 - b2 ** a["t"]
        for l in range(len(self.W)):
            for p, g, m, v in ((self.W, gW, a["mW"], a["vW"]), (self.b, gb, a["mb"], a["vb"])):
                m[l] = b1 * m[l] + (1 - b1) * g[l]
                v[l] = b2 * v[l] + (1 - b2) * g[l] ** 2
                p[l] -= lr * (m[l] / c1) / (np.sqrt(v[l] / c2) + eps)

  
    def state(self):
        d = {"sizes": np.array(self.sizes)}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            d[f"W{i}"], d[f"b{i}"] = W, b
        return d

    @classmethod
    def from_state(cls, d):
        m = cls(list(d["sizes"]))
        m.W = [d[f"W{i}"] for i in range(len(m.W))]
        m.b = [d[f"b{i}"] for i in range(len(m.b))]
        return m


def gradient_check(seeds=(0, 1, 2), eps=1e-5):
   
    worst = 0.0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        net = MLP([6, 5, 4, 3], seed=seed, l2=1e-2)
        net.b = [rng.normal(0, 0.1, b.shape) for b in net.b]
        X, y = rng.normal(size=(8, 6)), rng.integers(0, 3, 8)
        P, cache = net.forward(X)
        gW, gb = net.backward(y, cache)
        for params, grads in ((net.W, gW), (net.b, gb)):
            for l in range(len(params)):
                num = np.zeros_like(params[l])
                for idx in np.ndindex(params[l].shape):
                    old = params[l][idx]
                    params[l][idx] = old + eps; lp = net.loss(net.forward(X)[0], y)
                    params[l][idx] = old - eps; lm = net.loss(net.forward(X)[0], y)
                    params[l][idx] = old
                    num[idx] = (lp - lm) / (2 * eps)
                err = np.linalg.norm(num - grads[l]) / (np.linalg.norm(num) + np.linalg.norm(grads[l]) + 1e-12)
                worst = max(worst, float(err))
    return worst
