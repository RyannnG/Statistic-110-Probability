# Probobility 110

### Abbreviations  缩写

PDF - Probability Density Function 概率密度函数 $f(x)$

CDF -  Cummulative Distribution Function 分布函数 $F(X)$

PMF - Probabilty Mass Function 分布律 $P(X=k)$

R.V -  Random Variable 随机变量$X$  

Moment Generating Function**(MGF) 



## Lecture 1

Statistic is the logic of uncertainty

A **sample space** is the set of all possible outcomes of an experiment

An **event** is a subset of sample space

conterintuitive

Naive definetion of probability:

 P(A) = #favorable outcome / #possible outcomes

Example: file coins

Assumes all outcome equally likely, finite sample space

### Counting

Multip Rule:

if we have experiment with n~1~ possible outcomes, and  

### Sampling table

choose k objects out of n

|               | order matter     | order doesn't      |
| ------------- | ---------------- | ------------------ |
| replace       | n^k^             | $\binom{n+k-1}{k}$ |
| don't replace | n(n-1)...(n-k+1) | $\binom{n}{k}$     |

## Lecture 2

1. Don't lose common sense 
2. Do chek answers, esp by doing simple and extreme cases
3. Label people , objects etc. If have n people, then label them 1,2…n

Exp: 10 people, split into them of 6, team of 4 => $\binom{10}{6}$  2 teams of 5 => $\binom{10}{5}$ /2

Problem: pick k times from set of n objects, where order doesn't matter, with replacement.

Extreme cases: k = 0; k = 1; n = 2

Equiv : how many ways are there to put k indistinguishable particles into  n distinguishable boxes?



**Story proof- proof by interpretation**

Ex1 $\binom{n}{k}$ =  $\binom{n}{n-k}$ 

Ex2  n$\binom{n-1}{k-1}$ = k$\binom{n}{k}$  pick k people out of n, with one desigenate as president.

Ex3 $ \binom{m+n}{k} = \sum_{j=0}^k \binom{m}{j} \binom{n}{k-j} $ (vander minde) 

**Axioms of Probability- Non-naive definition**

probability sample consists of S and P, where S is sample space, and P , a function which takes an event A as input, returns P(A) and output. 

such that 

1. P() = 0, P(S) = 1
2. P(UAn) = sum of P(An) if A1,A2..An are disjoint (not overlap)

## Lecture 3

**Birthday Problem**

(Exclude Feb 29, assume 365 days equally likely, assume indep. of birth)

k people , find prob. that two have same birthday 

If k > 365, prob. is 1

50%-50% 

Let k <= 365, $P(no  match) =\frac{ 365 * 364 *... (365 - k + 1) }{365^k}$ 

P(match) ~ 50.7%, if k = 23; 97% if k = 50; 99.9999%, if k = 100

$\binom{k}{2} =  $       $\binom{23}{2} = 253$ 



**Properties of Probability**

1. P(A^*^) = 1 - P(A)    Proof: 1 = P(S) = P(AUA^*^) = P(A) + P(A^*^) 
2. If $A \subseteq B$ , then $P(A) \subseteq P(B)$ , Proof: $B = A\cup(B\cap A^*)$

$P(B) = P(A)+P(B\cap A^*)$ 

3. $P(A\cup B) = P(A) + P(B) - P(A\cap B)$, Proof: $P(A\cup B) = P(A\cap (B\cap A^*)) = P(A) + P(B\cap A^*)$ 

$P(A\cup B\cup C) = P(A) + P(B) + P(C) - P(A\cap B) - P(A\cap C) - P(B\cap C) + P(A\cap B\cap C)$ general case: 

deMortmort's Problem(1713), matching problem 

n cards labeled 1,2…n, let Aj be the event, ''jth card

P(Aj) = 1 / n since all position equally likely for card labeled j

P(A1\cap A2) = (n-2)! / n! = 1/n(n-1)

...

P(A1\cap … Ak) = (n-k)! / n!

P(A1\cup …An) = n*1/n - n(n-1)/2 * 1/n(n-1) + …= 1 - 1/2! + 1/3! - 1/4! … (-1)^n^ 1/n! ~ 1- 1/e

  ## Lecture 4

Define: Events A, B are indep. if $P(A\cap B) = P(A)P(B)$ 

Note: completely different form disjointness

A, B, C are indep, if P(A, B) = P(A)P(B), P(A,C) = P(A)P(C), P(B,C) = P(B)P(C) and P(A, B, C) = P(A)P(B)P(C)

Similarly for events A1,…An

Newton-Pepys Problem(1693)

Have fair dice; which is most likely?

(A) at least one 6 with 6 dice < - [ ] answer

(B) at least two 6 with 12 dice

(C) at least three 6 with 18 dice < Pepys believe

at least — union

P(A) = 1 - (5/6)^6^ ~ 0.665, P(B) = 1 - (5/6)^12^ - 12 *(1/6)(5/6)^11^    ~ 0.619

$P(C) = 1 - \sum_{k=0}^2 \binom{18}{k} (1/6)^k (5/6)^(18-k)$  ~ 0.597

**Conditional Probability** - How should you update prob./beliefs/uncertainty based on new evidence?

> "Conditioning is the soul of statistic" 

Define: $P(A|B) = P(A\cap B) / P(B)$, if P(B) > 0

Intuition 1: pebble world 9 pebbles , total mass is 1. P(A|B):get rid of pebbles in B^*^ , renormalize to make total mass again

Intuition 2: frequentist world: repeat ecperiment many times

100101101 001001011 11111111 

circle reps where B occurred ; among those , what fraction of time did A also occur?

**Theorem** 

1. $P(A\cap B) = P(B)P(A|B) = P(A)P(B|A)$
2. P(A1…An) = P(A1)P(A2|A1)P(A3|A1,A2)…P(An|A1,A2…An-1)
3. $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$ 

## Lecture 5

**Thinking** conditionally is a condition for thinking

How to solve a problem?

1. Try simple and extreme cases

2. Break up problem into simple pieces 

   P(B) = P(B|A1)P(A1) + P(B|A2)P(A2) +…P(B|An)P(An)

   law of total probability 

Ex get random 2 cards hand form standard deck

Find P(both aces|have ace), P(both aces|have ace of spade)

P(both aces|have ace) = P(both aces,~~have ace~~) / P(have ace) = (4,2)/(52, 2)/1 - (48, 2)/(52, 2) = 1/33

P(both aces|have ace of spade) = 3/51 = 1/17 

Patient get tested for disease afflicts 1% of population, tests positve 

Suppose test advertised as 95% accurate , suppose this mean 

D: has disease T: test positive

$P(T|D) = 0.95 = P(T^c |D^c)$

$P(D|T) = \frac{P(T|D)P(D)}{P(T)} = \frac{P(T|D)P(D)}{(P(T|D)P(D) + P(T|D^c)P(D^c)}$  

Biohazards

1. confusing P(A|B), P(B|A)  (procecutor's fallacy) 

<u>Ex</u> Sally Clark case, SIDS

want P(innocence |evidence) 

2. confusing P(A) - prior with P(A|B) - posterior 

P(A|A) = 1

3. confusing indep. with conditional indep. 

   Definition: Events A,B are conditionally indep. given C if P(A\cap B|C) = P(A|C)P(B|C)

   Does conditional indep given C imply indep? No

   Ex. Chess opponent of unknown strength may be that game outcomes are conditionally indep. given strength

   Does indep. imply conditional indep. given C? No

   Ex. A: Fire alarm goes off, cause by : F:fire; C:popcorn suppose F, C indep. But P(F|A, C^c^) = 1 not conditionally indep given A



## Lecture 6

**Monty Hall**

1 door has car, 2 doors have goats, Monty knows which 

Monty always open a goat door. If he has a choice , he picks with equal prob. Should you swich?

Note: if Monty opens door 2, we know door 2 has a goat, and Monty open ed door 2

LOTP:

wish we kenw where the car is

S: succeed (assuming switch)

Dj = Door j has car (j = 1, 2, 3)

P(S) = P(S|D1) 1/3 + P(S|D2) 1/3 + P(S|D3) /3

= 0 + 1 * 1/3 = 1* 1/ 3 = 2/3

By symmetry P(S|Monty opens 2) = 2/3

**Simpson's Paradox**

A: successful surgery

B: treated by Nick

C: heart surgery

P(A|B, C)  < P(A|B^c^,C)

P(A|B, C^c^) < P(A|B^c^, C^c^)

P(A|B) > P(A|B^c^)

C is a confounder

## Lecture 7

> Conditioning : the soul of statistics
>
> Random variables and their distribution 

**Gambler's Ruin**



Random walk: 

p = prob. of going right,  absorbing states 0, N

strategy: condition on first step

let Pi = (A wins game | A start with i dollars)

P~i~ = pp~i~+1 + qp~i-1~, 1 <= i <= N-1

(difference equation)

Guess P~i~ = X^i^ 

X^i^ = px^i+1^ + qx^i-1^ 

px^2^ - x + q = 0

x = {1, q/p}

p~i~ = AI^i^ + B(q/p)^i^

p~0~ = 0, B = -A, P~n~ = 1=> 1 = A(1-q/p)^n^ 

**Random Variable**

It's a function from sample space S to R

thnk of a as numerical "summary" of an aspect of the experiment.

**Bernoulli**

X is said to have Bern Distri. if X has only 2 possible  values , 0 and 1,

P(X=1) = p, P(X=0) = 1 - p. X = 1 is an event S:X(S) = 1

**Binomial (n,p) **

The distribution of #success X in n indep Bern(p) trials is called Bin(n, p) its 

distribution is given by 

$P(X=k) = \binom{n}{k}p^k (1-p)^{n-k} $ 

X ~ Bin(n,p), Y ~ Bin(m, p) indep.

Then X + Y ~ Bin(n + m, p) Proof consider n trials them m more trials

## Lecture 8

**Binnomial Distribution**

Bin(n, p)  X~Bin(n,p)

1. Story: X is #sucess in n **independent** Bern(p) trials

2. sum of indicator : X=X1 + X2 +…+Xn, Xj = 1 if jth trial success ,0 otherwise   

   i.i.d.Bern(q)  => indep. identically distributed

3. **PMF** $P(X=k) = \binom{n}{k}p^k (1-p)^{n-k}$  

**PMF -Probabilty Mass Function**

**R.V** random variable

X = 7 is an event

**CDF** 

X<= x is an event

F(x) = P(X<=x)

then F is the CDF of X (**cummulative distribution function**) 

**PMF**(for discrete  r.v.s)  

Discrete: possible values a1, a2, …an could be listed out

P(X=aj) for all j = pj

pj >= 0, sum of all pj = 1



X~Bin(n,p), Y~Bin(m,p) => X+Y ~Bin(n+m, p)

1. immediate form story

2. X = x1 +…+ xn, Y = y1 +…+ yn => X+Y = sum of xs + sum of ys

   sum of n+m i.i.d Bern(p) => Bern

3. $P(X+Y=k) = \sum_{j=0}^k P(X+Y=k|X=j)P(X=j) $ 

   $ = \sum_{j=0}^k P(Y=k-j|X=j) \binom{n}{j}p^j q^{n-}j$   

   $= \sum_{j=0}^k \binom{m}{k-j} p^{k-j}q^{m-k-j} \binom{n}{j} p^j q^{n-j}$

   $=p^k q^{m+n-k} \sum_{j=0}^k \binom{m}{k-j} \binom{n}{j}$ 

   VanderMorde $\sum_{j=0}^k \binom{m}{k-j} \binom{n}{j} = \binom{m+n}{k}$

   convolution

Ex. 5 card hand find distribution of #aces  - PMF(or CDF)

let X = (#aces)  find P(X=k),  Not Binomial .  Like the elk problem

**Hypergeometric**

Have b black, w white marbles. Pick simple random sample of size n. Find list. of (#write marbles in sample) = X

$$P(X=k) =\frac{\binom{w}{k} \binom{b}{n-k}}{\binom{w+b}{n}}$$ **Hypergeometric** sampling without replace 

$$\frac{1}{\binom{w+b}{n}} \sum_{k=0}^w \binom{w}{k} \binom{b}{n-k} = 1$$  

CDF P(X<=x)

## Lecture 9

**CDF**

F(x) = P(X<=x), as a function of real x

P(a<X<b) = F(b) - F(a)

Peoperties of CDF

1. increasing
2. right continuous 
3. F(x) -> 0 as x -> $- \infty$  F(x)-> as x -> $\infty$ 

This is "only if"

**Indep. of r.v.s**

X,Y are indep. r.v.s if P(X<=x, Y<=y) = P(X<= x)P(Y<=y) for all x, y

Discrete case : P(X=x, Y=y) = P(X=x)P(Y=y)

**Average(Means, Expected Values)**

1,2,3,4,5,6, -> 1+2+..+6 / 6 = 3.5

1,1,1,1,1,3,3,5

two ways: 1. add, divide 2. 5/8 * 1 +2/8 * 3 + 1/8 * 5

**Average of a discrete r.v.s**

$E(X) = \sum_{} xP(X=x)$ 

**X~Bern(p) **

$E(x) = 1P(X=1) +0P(X=0) = p$

X = 1 if A occurs , 0 otherwise (indicator r.v.s)

$E(x) = P(A)$  fundamental bridge

**X~Bin(n,p)**

$E(X) = \sum_{k=0}^n k\binom{n}{k} p^k q^{n-k}$ 

$=  \sum_{k=1}^n n \binom{n-1}{k-1} p^k q^{n-k}$

$ = np \sum_{k=0}^n  \binom{n-1}{k-1} p^{k-1} q^{n-k}$ 

$ = np \sum_{j=0}^{n-1}  \binom{n-1}{j} p^{j} q^{n-j-1}$ 

$=np$  

**Linearity**

$E(X+Y) = E(X) +E(Y)$  even if X, Y are dependent

$E(cX) = cE(X)$ 

**Redo Bin**

np  by linearity since X = x1+..+xn 

**Ex**. 5 Card hand ,X = #aces let Xj be indecator of jth card being as ace, 1<=j <=5

E(X) =(indicator) E(X1 + ..+X5) =(linearity) E(X1) +..+ E(X5) = (symmetry) 5E(X1)

= (fundamental bridge) 5P(1st card  ace) = 5/13, even though Xj's are dependent

This gives expected value of any Hyoergeometric

**Geometric **

Geom(p) : indep. Bern(p) trials, count #failures before 1st success. let X~Geom(p) q = 1-p

PMF: $P(X=k) = q^k p$  valid since $\sum_{k=0}^{\infty} pq^k = p/1-q = 1$

  $E(x) = \sum_{k=0} kpq^k$

$=p \sum_{k=1} kq^k$

$= q/p$

story proof: let c = E(X), $c = 0*p + (1+c)q$

$= q+cq => q/p$ v

## Lecture 10

**Linearity**

Let T = X+Y, show E(T) = E(X)+E(Y) 

$\sum_{t} P(T=t) = \sum_{x} xP(X=x) + \sum_{y} yP(Y=y)$  

Extreme dependent X=Y

E(X+Y) = E(2X) =2E(X)

**Negetive Binomial**

parameters r,p

story: indep. Bern(p) trials #failures before the rth success

PMF: $P(X=n) =   \binom{n+r-1}{r-1} p^r (1-p)^n$   

$E(X) = E(X1+..+Xr) = E(X1) +..+E(Xr) = rq/p

Xj is #failures between (j-1)th and jth success, Xj~Geom(p) 

**Geom**

X~FS(p) time until 1st success , counting the success

Let Y = X-1, Then Y~Geom(p) 

E(X) = E(Y+1) = E(Y) + 1 =q/p + 1 = 1/p

**Putnam**

Random permutation of 1,2,..n , where  n>= 2

Find expected # of local maxima.  Ex. **3**214**7**5**6**  

Let I~j~  be indecator r.v of position j having a local max, 1<=j <=n

$E(I1 + ..In) = E(I1)+..+E(In) = \frac{n-2}{3}  + 2/2 = \frac{n+1}{3}$   

**St.Petersburg Paradox**

Get $2^x^ where X is #filps of fair coin until first H, including the success

Y = 2^x^ find E(Y) 

$E(Y) = \sum_{k=1}^{\infty} 2^k \frac{1}{2^k} = \sum_{} 1 = \infty$   

bound at $$2^{40}$$  .  Then $\sum_{k=1}^{40} 2^k \frac{1}{2^k} = 40$   

E(2^x) =\infty not q= 2^{E(x)} = 4

## Lecture 11

**Sympathetic magic**

Dont' confuse r,v with its distribution

~~P(X=x) + P(Y=y)~~ 

> Word is not the thing,  the map is not the territory.

r.v -> random house  distribution -> blueprint

**Poisson Distribution** $X \sim Pois(\lambda)$

PMF: $P(X=k) = e^{-\lambda} \frac{\lambda^k}{k!} $   \lambda is the rate parameter >0

Valid: $\sum_{k=0}^{\infty}e^{-\lambda} \frac{\lambda^k}{k!} = 1$

$E(X) = \lambda e^{-\lambda}\sum_{k=1}^{\infty} \frac{\lambda^{k-1}}{(k-1)!} = \lambda $  

often used for applications where counting # of "successes" where ther are a large  # trials  each with small prob of success

1. \#emails in an hour


2. \#chips in choc chip cookies
3. \#earthquakes in a year in some area

**Pois Paradigm**

Events A1, A2,..An, P(Aj) = pj, n large, pj's small

events indep. or "weakly dependent"  

\# of  Aj's that occure is approx Pois(\lambda) \lambda = sum of pj

Binormial converges to Poisson

Raindrops

Ex. Have n people, find approx  prob that there are 3 people with same birthday.

 $\binom{n}{3}$ triplets of people , indicator r,v for each, I~ijk~ , i<j<k

$E(triple matches) = \binom{n}{3}1/365^2$  

X = #triple matches  Approx Pois(\lambda) $\lambda = \binom{n}{3} 1/365^2$ 

I123, I124 are not indep. 

P(X>=1) = 1 - P(X=0) ~ $1 - e^{-\lambda}$ 

## Lecture 12

**Discrete**

X

PMF P(X=x)

CDF F~x~(x) = P(X,=x)

$E(x) = \sum_{} xP(X=x)$

Var(X) = 

LOTUS $E(g(x)) = \sum_{} g(x)P(X=x)$

**Continous**

X

PDF f~x~(x) = F~X~'(x)

CDF F~X~(x) = P(X<=x)

any specific value P(X=x) = 0

**PDF - Probability Density Function**

Defn: R.v X has PDF f(x) if $P(a\leq X\leq b) = \int_{a}^{b} f(x)dx $  for all a and b

to be valid f(x) >= 0, intergal of f(x) = 1

If X has PDF f, the CDF is $F(x) = P(X\leq x) = \int_{-\infty}^{x}f(t)dt $   

If X has CDF F ( and X is continous ), then f(x) =F'(x) by FTC

$P(a\le X\le b) = \int_{a}^{b} f(x)dx = F(b) - F(a )$   

$E(X) = \int xf(x)dx$ 

**Variance**

$Var(X) = E(X - EX)^2$  

**Standard deviation**: $SD(X) = \sqrt{Var(X)}$  

Another Way to Express Var:

$Var(X) = E(X^2 - 2X(EX) + (EX)) = E(X^2) - 2E(X)E(X) + (EX)^2 = E(X^2) - (EX)^2$

Notation $EX^2 = E(X^2)$ 

**Uniform** Unif(a,b)

prob. length

f(x) = c, if a <= x <= b; 0, otherwise 

$1 = \int_{a}^b cdx = c = \frac{1}{b-a}$ 
$$
F(X)= \int_{-\infty}^{x} f(t)dt = 0, if x < a; 1, if x > b; x-a/x-b, if a<=x <= b;
$$
$E(X) = \int_{a}^{b} \frac{x}{b-a} dx = \frac{a+b}{2} $  

$Y = X^2, E(X^2) =E(Y) = \int_{-\infty}^{\infty} x^2 f(x)dx $ 

law of the unconscious statistician (LOTUS)

$E(g(X)) = \int_{-\infty}^{\infty} g(x)f(x)dx $  

Var(X) = 

Let U~Unif(0,1) E(U) = 1/2 ,$ E(U^2) = \int_{0}^{1} u^2f(u)du = 1/3$

Var(U) = 1/3 - 1/4 = 1/12

> Uniform is Universal 

let U~Unif(0,1) F be CDF (assume F is strictly increasing and continuous)

Then Let $X = F^{-1}(U)$, Then X~F

Proof:  $P(X\leq x) = P(F^{-1} (U) \leq x) =P(U\leq F(x)) = F(x)$ 

## Lecture 13

**Universality of Unif** 

let  F be a cont. strictly increasing CDF

Then $X = F^{-1}(U)$ ~ F if U~unif(0, 1)

Also: if X~F, then F(X)~Unif(0,1)

$F(x) = P(X\leq x)$ ~~F(X) = P(X \leq X) = 1~~ 

Ex. Let  $F(x) = 1 - e^{-x}, x>0$  (Expo(1)), U~Unif(0,1)

simulate X~F. $F^{-1}(u) = -\ln (1-u)$  => $F^{-1}(U) = -\ln (1-U) \sim F $ 

$1-U \sim Unif(0, 1)$ symmetry of Unif

$a+bU$ is Unif on some interval. Nonlinear usually -> Non Unif. 

**Indep. of r.v.s** X1,..Xn

Defn X1,…Xn indep if P(X1 <= x1, …Xn <= xn) = P(X1<=x1)…P(Xn<=xn)

for all x1,…xn

Discrete case: joint PMF P(X1 = x1,..Xn=xn) = P(X1-x1)…P(Xn=xn) 

Ex. X1, X2 ~ Bern(1/2) i.i.d, X3 = 1 if X1 = X2; 0 otherwise

These are pairwise indep, not indep

**Normal Distribution**

(Central Limit Therom: sum of a lot of i.i.d r.v.s looks Normal )

$N(0, 1)$ - mean = 0, var = 1 

has PDF  $$f(z) = ce^{\frac{z^2}{2}}$$  c - normalizing const

$c = 1/\sqrt{2\pi}$ 

$Z\sim N(0,1)$ 

EZ = 0 by symmetry odd function

E(Z^3) = 0 "3rd moment"

$ Var(Z) = E(Z^2) - (EZ)^2 = E(Z^2) = 1$  LOTUS

Notation : $\Phi$ is the standard Normal CDF 

 $$\Phi(z) = \frac{1}{\sqrt2\pi} \int_{-\infty}^z e^{-\frac{t^2}{x}}dt$$  

$\Phi(-z) = 1- \Phi(z)$  by symmetry

## Lecture 14

-Z~N(0,1) (symmetry) Normal

Let $X = \mu + \sigma Z$  $\mu$ (mean, location) $\sigma >0$ (SD, scale) 

Then we say $X \sim N(\mu, \sigma^2)$ 

$E(X) = \mu, Var(X) = \sigma^2Var(Z)$ 

$Var(X+c) = Var(X)$ 

$Var(cX) = c^2 Var(X)$ 

$Var(X) \geq 0, Var(X) = 0$, if and only if P(X=a) = 0, for some a

$Var(X +Y) \ne Var(X) + Var(Y)$  in general , Var not linear

[equal if X,Y are indep.]

$Var(X+X) = Var(2X) = 4Var(X)$ 

$Z = \frac{X-\mu}{\sigma}$  standard 

Find PDF of $X \sim N(\mu, \sigma^2)$ 

CDF: $P(X\le x) = P(\frac{X-\mu}{\sigma} \le \frac{x-\mu}{\sigma}) = \Phi(\frac{X-\mu}{\sigma})$ 

$-X = -\mu + \sigma(-Z) \sim N(-\mu, \sigma^2)$ 

Later we'll show if Xj ~ N(\mu, sigma j^2) indep, 

$X_1 + X_2 \sim N(\mu_1 + \mu_2, \sigma_1^2+\sigma_2^2)$ 

$X_1 - X_2 \sim N(\mu_1 - \mu_2, \sigma_1^2+\sigma_2^2)$ 

**68-95-99.7% Rule**

**LOTUS**

X: 0, 1, 2, 3...

X^2^: 0, 1, 4 ,9...

$E(X) = \sum_{x} P(X=x)$

$E(X^2) = \sum_{x} x^2 P(X=x)$ 

$X \sim Pois(\lambda)$

$E(X^2) = \sum_{k=0}^{\infty}k^2 e^{-\lambda} \frac{\lambda^k}{k!}  = \lambda^2 + \lambda$ 

$Var(X) =  \lambda^2 + \lambda - \lambda^2 = \lambda$

$\sum_{k=0}^{\infty} \frac{\lambda^k}{k!} = e^{\lambda}$ always true

$\lambda \sum_{k=1}^{\infty} \frac{k \lambda^{k-1}}{k!} =\lambda e^{\lambda}$ 

$\sum_{k=1}^{\infty} \frac{k^2 \lambda^{k-1}}{k!} =(\lambda+1) e^{\lambda}$ 



$X\sim Bin(n,p)$ Find Var(X)

$X = I_1 + ..I_n, I_j \sim Bern(p)$ 

$X^2 = I_1^2 +..+ I_n^2 + 2I_1I_2+..+ 2I_{n-1}I_n$ 

$E(X^2) = nE(I_1^2) + 2\binom{n}{2}E(I_1I_2)$  indicator of success on both trials 1,2

$= np + n(n-1)p^2 = np + n^2p^2 - np^2$

$Var(X) = np - np^2 = np(1-q) = npq, q=(1-p)$   



**Prove LOTUS for discrete sample space**

$E(g(x)) = \sum_{} g(x)P(X=x)$ 

 group $\sum_{x} g(x)P(X=x) = \sum_{s\in S} g(X(s))P(\{s\})$   ungrouped

## Lecture 15

**Coupon collector**(Toy collector)

n toy types, equally likely,  find expected time   until have complete set

T = T_1 + T_2 + .. T_n

T_1 =(time until 1st new toy) = 1

T_2 = (Addtional time until 2nd new toy)

T_3 = (…..until 3rd )

T_1 = 1

$T_2 - 1 \sim Geom(n-1/n)$ 

$T_j - 1 \sim Geom(\frac{n-(j-1)}{n})$ 

$E(T) = E(T_1) + E(T_2) +… E(T_n) = 1 + n/(n-1)+ n/(n-2) + n/1 = n(1 + 1/2 +…+ 1/n) \approx n\log n$  

**Universality**

$X\sim F$

$F(x_0) = 1/3$

$P(F(X) \le 1/3) = P(X \le x_0) = F(x_0) = 1/3$ 

**Logistic Distribution**

$$F(x) = \frac{e^x}{1 + e^x}$$

$U\sim Unif(0, 1)$ , consider $F^{-1}(U) = \log \frac{U}{1-U}$ is logistic



Let X, Y, Z ,be i.i.d positive r.v.s  Find $E(\frac{X}{X+Y+Z})$ 

$E(\frac{X}{X+Y+Z})  = E(\frac{Y}{X+Y+Z}) = E(\frac{Z}{X+Y+Z})$ by symmetry 

$E(\frac{X}{X+Y+Z})  + E(\frac{Y}{X+Y+Z}) + E(\frac{Z}{X+Y+Z}) = E(\frac{X+Y+Z}{X+Y+Z}) = 1$ by linearity

$E(\frac{X}{X+Y+Z}) = 1/3$ 

**LOTUS**

$U\sim Unif(0,1)$  $X=U^2, Y=e^x$

Find E(Y) as an integral

$E(Y) = \int_{0}^{1}e^{x}f(x)dx$  f(x) PDF of x,  need more work

$P(U^2 \le x) = P(U\le \sqrt{x}) = \sqrt{x},0 <x<1$ 

​ 

Better: $Y = e^{U^2}$  $E(Y) = \int_{0}^{1}e^{U^2}dU$  



$X\sim Bin(n,p), q =1-p$

find distribution of n-X

$P(n-X=k) = P(X=n-k) = \binom{n}{n-k}p^{n-k}q^k = \binom{n}{k}q^kp^{n-k}$ 

story: $n-X \sim Bin(n,q)$ 

by swapping ''success and failure" 



Ex #emails I get in time t is $Pois(\lambda t)$ 

Find PDF of T, time of 1st  email.

$P(T>t) = P(N_t = 0) = e^{-\lambda t}(\lambda t)^0 / 0! = e^{-\lambda t}$ 

with N_* = (#emails in [0, *])

CDF is $1 - e^{-\lambda t}, t>0$ 



Distribution is the blueprint, for creating random variable, that was our random house and then don't confuse random variable with a constant 

constant would be like a specific house

the random vairable is that the random house

$f(x) = \frac{1}{2}x^{-\frac{1}{2}}, x\in (0,1)$ 

## Lecture 16

#### Exponential Distribution

rate parameter $\lambda$

$X \sim Expo(\lambda)$ has PDF $\lambda e^{-\lambda x}, x>0$ 0otherwise

CDF $F(x)= \int_{0}^{x}\lambda e^{-\lambda t}dt = 1 - e^{\lambda x}, x>0$ 

Let $Y= \lambda X$ then $Y\sim Expo(1)$ 

since $P(Y\le y) = P(X\le y/\lambda)= 1- e^{-y}$ 

Let $Y\sim Expo(1)$ find E(Y), Var(Y)

$E(Y) = \int_{0}^{\infty} ye^{-y}dy = 1$ , du = dy, dv = -e^-y^

$Var(Y) = E(Y^2) - (EY)^2 = 1$  LOTUS

So $ X=Y/\lambda$ has $E(X) = 1/\lambda, Var(X) = 1/\lambda^2$ 

**Memoryless Property** 

$P(X\ge s+t|X\ge s) = P(X\ge t)$ 

Here $P(X\ge s) = 1  -P(X\le s) = e^{-\lambda s}$

$P(X\ge s+t|X\ge s) = P(X\ge s+t, X\ge s) / P(X\ge s) =  P(X\ge s+t) / P(X\ge s)  = e^{-\lambda t} = P(X\ge t)​$ 

$X\sim Expo(\lambda)$

$E(X|X>a) = a + E(X-a|X>a) = a + q/\lambda$  by memoryless

## Lecture 17

$E(T|T>20) > E(T)$  

IF memoryless, we would have E(T|T>20) = 20 + E(T)

Therom: IF X is a positive continuous r.v.  with memoryless property, then $X\sim Expo(\lambda)$  for some $\lambda$ 

**Proof** Let F be the CDF of X, G(x) = P(X>x) = 1 - F(x)

memoryless property is G(x+t) = G(s)G(t) solve for G.

let s=t, $G(2t) = G(t)^2$ , $G(3t) = G(t)^3$ …$G(kt) = G(t)^k$ 

$G(t/2) = G(t)^{1/2}$ …$G(t/k) = G(t)^{1/k}$

$G(\frac{m}{n} t) = G(t)^{\frac{m}{n}}$  So $G(xt) = G(t)^x $ for all real x >0

let t = 1, $G(x) = G(1)^x = e^{x\ln G(1)} = e^{-\lambda x}$  $ lnG(1) = -\lambda$ 

**Moment Generating Function**(MGF)

Defn ;A r.v X has MGF $M(t) = E(e^{tx})$ 

as a function of t, if this is finite on some (-0, a), a>0

t is just a placeholder

Why moment "generating" ?

$$E(e^{tx}) = E(\sum_{n=0}^{\infty} \frac{x^n t^n}{n!}) = \sum_{n=0}^{\infty}\frac{E(x^n)t^n}{n!}$$  $E(x^n)$ - nth moment



Three reasons why MGF important: 

Let X have MGF M(t)

1. The nth moment $E(x^n)$ , is coef of $\frac{t^n}{n!}$  in Taylor series of M, 

   and    $M^{(n)}(0) = E(X^n)$ 

2. MGF determines the distribution.  i.e. if X,Y have same MGF , then they have same CDF

3. If X has MGF M_x, Y has MGF M_y, X indep. of Y, then MGF of X+Y is $E(e^{t(X+Y)}) = E(e^{tX})  + E(e^{tY})$ 

Ex. $X\sim Bern(p)$, $M(t) = E(e^{tX}) = pe^t + q, q= 1-p$

$X\sim Bin(n, p) => M(t) = (pe^t + q)^n$ 

$Z\sim N(0,1)$  => $$M(t) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty}e^{tZ - \frac{Z^2}{2}}dz$$ 

$$= \frac{e^{t^2/2}}{\sqrt{2\pi}}\int_{-\infty}^{\infty}e^{-\frac{1}{2}(Z-t)^2}dz = e^{t^2/2}$$

**Laplace Rule of Succession**

Given p,  X1, X2 … i.i.d,  Bern(p)

p unknown, 

Laplace used the rule of succession to calculate the probability that the sun will rise tomorrow

Bayesian : treat p as a r.v.

Let $P\sim Unif(0,1)$  (prior) Let S_n = X1,…Xn

So$ S_n|p \sim Bin(n,p)$ , $p \sim Unif(0,1)$

Find Posterior $p|S_n$ , and $P(X_{n+1} = 1| S_n = n)$ 

$$f(p|S_n = k) = \frac{P(S_n = k|p)f(p)}{P(S_n =k)}$$  

 f(p) - prior, 1;  P(S_n =k) does not depend on p

$P(S_n = k) = \int_{0}^{1}P(S_n = k/p)f(p)dp$ 

$$ \varpropto \frac{p^k(1-p)^{n-k}}{f(p|S_n = n)=(n+1)p^n}$$

$p(X_{n+1}=1|S_n = n) = \int_{0}^{1}(n+1)pp^mdp= \frac{n+1}{n+2}$ 



## Lecture 18

**Expo MGF**

$X\sim Expo(1)$, find MGF, moment 

$M(t) = E(e^{tx}) = \int_{0}^{\infty} e^{-tx}e^{-x}dx = \int_{0}^{\infty} e^{-x(1-t)}dx = \frac{1}{1-t}, t<1$ 

M'(0) = E(X), M"(0) = E(X^2^) , M'''(0) = E(X^3^)….

$ |t| <1,  \frac{1}{1-t} = \sum_{n=0}^{\infty}t^n = \sum_{n=0}^{\infty} n!\frac{t^n}{n!} $    $E(X^n) = n! $

$Y\sim Expo(\lambda)$, let $X = \lambda Y \sim Expo(1)$, so $Y^n = \frac{X^n}{\lambda^n}$ 

$E(Y^n) = \frac{E(X^n)}{\lambda^n} = \frac{n!}{\lambda^n}$ 

**Normal MGF**

Let $Z\sim N(0,1)$ ,find all its moment

$E(Z^n) = 0$ for n odd by symmetry

MGF $$M(t) =e^{t^2/2} = \sum_{n=0}^{\infty} \frac{(t^2/2)^n}{n!} =  \sum_{n=0}^{\infty} \frac{(2n)!t^{2n}}{z^n n!(2n)!}$$  

=>$$E(Z^{2n}) = \frac{(2n)!}{2^n n!}$$ 

**Poisson MGF**

$X\sim Pois(\lambda)$   $E(e^{tx}) = \sum_{k=0}^{\infty} e^{tx} e^{-\lambda} \frac{\lambda^k}{k!} = e^{-\lambda} e^{\lambda e^t} $ 

let$ Y\sim Pois(\mu)$ **indep** of X ,find distribution  of X+Y , 

Multiply  MGFs, $e^{\lambda(e^t-1)}e^{\mu (e^t-1)} = e^{(\lambda+\mu)(e^t-1)}$  => $X+Y \sim Pois(\lambda + \mu)$ 

sum of independent Poisson is still Poisson

 Counterexample if X, Y dependent: X =Y => X+Y = 2X is not Poisson since even;

$E(X+Y) = E(2X) = 2\lambda, Var(2X) = 4\lambda $ 

**Joint Distribution**

X, Y Bernouli

Ex. 2D

|      | Y=0  | Y=1  |      |
| ---- | ---- | ---- | ---- |
| X=0  | 2/6  | 1/6  | 3/6  |
| X=1  | 2/6  | 1/6  | 3/6  |
|      | 4/6  | 2/6  |      |

They are indep. 



X, Y r.vs 

**joint CDF**

$F(x,y) =P(X\le x, Y \le y)$

**joint PMF (discrete case)**

$P(X=x, Y=y)$ 

**Marginal CDF**

$P(X\le x)$ is marginal dist. of X

**Joint PDF (cont.)**

f(x, y) such that 

$P((X,Y)\in B) = \iint_{B} f(x, y)dxdy$ 

**independence**

X,Y indep. if and only if $F(x,y) = F_X(x)F_Y(y)$ 

Equiv. 

$P(X=x, Y=y) = P(X=x)P(Y=y)$

$ f(x, y) = f_X(x)f_Y$ for all x, y >0

**Getting marginals**

$P(X=x, Y=y) = \sum_y P(X=x, Y=y)$ discrete

$f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x,y)dx$

Ex. Uniform on $square\{(x,y): x,y\in [0,1]\} $

joint PDF const. on the square, 0 outside

integral is area c = 1/area = 1

marginal:  X,Y are indep. Unif(0,1)

Ex Unif in disc $x^2 = y^2 \le 1$ 

joint PDF: $1/\pi$ inside the circle; 0 otherwise

X,Y dependent. 

Given X=x , $|y| \le \sqrt{1-x^2}$ 

## Lecture 19

Joint , conditional, marginal, dist

joint CDF $F(x,y) = P(X\le x, Y\le y)$ 

cont. case(jonit PDF) : $f(x, y) = \frac{\partial}{\partial x \partial y}F(x, y)$ 

$P((X, Y) \in A) = \iint_{A} f(x,y)dxdy$ 

marginal PDF of X: $\int f(x,y)dy$ 

conditional PDF of Y|X is 

$f_{Y|X}(y|x) = f_{X,Y}(x,y) / f_{X}(x)  = f_{X|Y}(x|y)f_{Y}(y)/f_{X}(x) $

X, Y indep. if $f_{X,Y}(x,y) = f_X(x)f_Y(y)$ for all X,Y



**2-D LOTUS**

Let (X,Y) have joint PDF f(x,y)

and let g(x, y) be a real-valued fn of x,y

Then $Eg(X,Y) = \iint g(x,y)f(x,y)dxdy$ 

Theorm

IF X,Y are indep, then E(XY) = E(X)E(Y)

Indep. implies uncorrelated

Proof (continuous case)

$E(XY) = \iint xyf_X(x)f_Y(y)dxdy =  \int yf_Y(y) \int xf_X(x)dxdy = (EX)(EY)$ 

Ex. X, Y i.i.d Unif(0,1) find E|X-Y|

 LOTUS $\int_0^1 \int_0^1 |x-y|dxdy = \iint_{x>y}(x-y)dxdy + \iint_{x\le y} (y-x)dxdy$ 

$= 2\int_0^1 \int_y^1 (x-y)dxdy = 2\int_0^1 (x^2/2 - yx)|_y^1 dy = 1/3$ 

Let M = max(X,Y)

L = min (X, Y)  (L stand for little and less one not large one)

|X-Y| = M-L

E(M-L) = 1/3

E(M)-E(L) = 1/3

E(M+L)= E(X+Y) = E(M)+E(L) = 1

=> E(M) = 2/3, E(L) = 1/3

**Chicken-egg**

some hens some hatch some don't hatch, the eggs are indep.

N eggs,  $N \sim Pois(\lambda)$, each hatches prob p, indep, Let X = #hatch

so $X|N \sim Bin(N,p)$ 

 Let Y = # don't hatch, so X + Y = N

Find joint PMF of X,Y

$P(X=i, Y=j) = \sum P(X=i, Y=j| N=n)P(N=n) $

$= P(X=i, Y=j|N=i+j)P(N=i+j)$ 

$$ = P(X=i|N=i+j)P(N=i+j) =\frac{(i+j)}{(i!j!)} p^i q^j \frac{e^{-\lambda} \lambda^{i+j}}{(i+j)!} $$  

$ = (e^{\lambda p} \frac{(\lambda p)^i}{i!}) (e^{\lambda q} \frac{(\lambda q)^j}{j!})$ 

=> X, Y are indep, $X\sim Pois(\lambda p),  Y\sim Pois(\lambda q)$ 

## Lecture 20

Ex FInd $E|Z_1 - Z_2|$, with Z1, Z2 i.i.d N(0,1)

Therom 

$X\sim N(\mu_1, \sigma_1^2), Y\sim N(\mu, \sigma_2^2)$ indep

Then $X+Y \sim N(\mu_1+\mu_2, \sigma_1^2 + \sigma_2^2)$ 

Proof 

Use MGF, MGF of X+Y is 

 $$e^{\mu_1t +\frac{1}{2} \sigma_1^2 t^2}  e^{\mu_2t +\frac{1}{2} \sigma_2^2 t^2}$$ 

Note Z1 Z2 ~ N(0, 2)

$E|Z1-Z2| = E|\sqrt{2} Z|$ Z~N(0, 1)

$= \sqrt{2}E|Z| = \sqrt{2/\pi}$ 

**Multinomial** 

generalization of binomial

Defn/story of $Mult(n,\vec{p})$ , 

$\vec{p} = (p_1,…p_k)$ prob. vector$ p_j \ge 0, \sum pj = 1$

$\vec{X}\sim Mult(n, p), X = (X_1, … X_k)$ 

have n objects indep. putting into k categories

$P_j = P(category j)$ $X_j$ = #objects in category j

Joint PMF $P(X_1 = n_1, ..X_k = n_k) = \frac{n!}{n_1!n_2!…n_k!} P_1^{n_1} P_2^{n_2}...P_k^{n_k} $ 

if$n_1 +..+ n_k  = 1$; 0 otherwise

$\vec{X}\sim Mult(n,p)$  Find marginal dist of  $X_j$ Then $X_j \sim Bin(n, p_j) $ 

(each of objects either in this category j or it isn't)

$E(X_j) = np_j, Var(X_j)= np_j(1-p_j)$

**Lumping Property**

Merge category together

$\vec{X} = (X_1, … X_10) \sim Mult(n, (p_1,…p_10))$ 

ten political parties, take n people , ask people which party they in 

$\vec{Y} = (X_1, X_2, X_3 + ..+ X_{10})$  Then $Y \sim Mult(n, (p_1, p_2,p_3+..+p_{10}))$

(wouldn't work if one can be in more than one category)

$\vec{X}\sim Mult(n, p)$, Then give $X_1 = n_1$ ,  PMF 

$(X_2,…X_k) \sim Mult_{k-1}(n-n_1, (p'_2,…p'_k))$ 

(we know how many people in the first catgory , don't know rest)

with $p'_2$ = P(being in category 2| not in category 1) 

= $\frac{p_2}{1-p_1}$  

$$p'_j = \frac{p_j}{p_2+…p_k}$$  

**Cauchy Interview Problem**

The Cauchy is dist. of T = X/Y with X, Y i.i.d N(0,1)

Find PDF of T

(doesn't have a mean and variance)

average of million cauchy is still cauchy

$P(\frac{X}{Y} \le t) = P(\frac{X}{|Y|} \le t)$  symmetry of N(0.1)

$= P(X\le t|Y|) = \frac{1}{\sqrt{2\pi}} \int_{\infty}^{\infty} e^{y^2 /2} \int_{-\infty}^{t|y|} \frac{1}{\sqrt{2\pi}} e^{x^2 /2}  dxdy $ 

$= \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} e^{y^2 /2} \Phi(t|y|)dy$ 

$ = {\sqrt{2/\pi}} \int_{0}^{\infty} e^{y^2 /2} \Phi(ty)dy $ 

PDF: $F'(t) = 1/\pi(1+t^2)$ 

$P(X\le t|Y|) = \int P(X\le t|Y|| Y=y)\varphi(y)dy$

$= \int \Phi(t|y|)\varphi(y)dy$ 

## Lecture 21

Define $Cov(X,Y)= E((X-EX)(Y-EY))$ 

$= E(XY) - E(X)E(Y)$ 

since =E(XY)-E(X)E(Y) - E(X)E(Y) + E(X)E(Y)

Property 

1. $Cov(X,X) = Var(X)$

2. $Cov(X,Y) = Cov(Y,X)$ 

3. $Cov(X,c) = 0$, if c is const.

4. $Cov(cX,Y) = cCov(X,Y)$  

5. $Cov(X,Y+Z)=Cov(X,Y) + Cov(X,Z)$ 

   bilinearty(4 , 5)

6. $Cov(X+Y, Z+W) = Cov(X,Y) + Cov(X,W)+Cov(Y,Z)+Cov(Y,W)$

   $Cov(\sum_i^m a_iX_i, \sum_j^n b_j Y_j) = \sum_{i,j} a_i b_j Cov(X_i, Y_j)$  

7. $Var(X_1 + X_2)=Var(X_1) + Var(X_2) + 2Cov(X_1, X_2)$ 

   $Var(X_1+…+ X_n) = Var(X_1)+..+ Var(X_n) + 2\sum_{i<j} Cov(X_i,X_j)$  



Therom 

If X, Y are indep. then they're uncorrelated i.e Cov(X,Y) = 0

Converse is false(common mistake) 

e.g. $Z\sim N(0.1)$ 

$X=Z, Y=Z^2, Cov(X,Y) = E(XY) - E(X)E(Y) = E(Z^3) - E(Z)E(Z^2) = 0$

but very dependent: Y is a function of X, (we know X then we know Y)

Y determines magnituide of X

Define $Corr(X,Y) = \frac{Cov(X,Y)}{SD(X)SD(Y)} = Cov(\frac{X-EX}{SD(X)}, \frac{Y-EY}{SD(Y)})$ 

Therom 

$-1 \le Corr \le 1$  (form of Cauchy-schwarz)

Proof 

WLOG  assume X,Y are standardized  let $Corr(X,Y) = \rho$

$0 \le Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y) = 2 + 2\rho$

$0 \le Var(X-Y) = Var(X) + Var(Y) - 2Cov(X,Y) = 2 - 2\rho$

= > $0 \le \rho \le 1$ 

Ex. Cov in a Multinomial 

(X_1, … X_k) \sim Mult(n, \vec{p})

Find Cov(X_i, X_j) for all i,j

If i=j, $Cov(X_i, X_i) = Var(X_i) = np_i(1-p_i)$

Now let $i \ne j$

fiind Cov(X_1, X_2) = c, 

$Var(X_1 + X_2) = np_1(1-p_1) + np_2(1-p_2) + 2c$

$= n(p_1+p_2)(1-(p_1+p_2))$ => c = -np_1p_2

General: $Cov(X_i, X_j) = -np_ip_j, for i\ne j$

Ex. X\sim Bin(n,p), write as X=X_1+…X_n  X_j are i.i.d Bern(p)

$Var X_j = EX_j^2 - (EX_j)^2 = p - p^2 = p(1-p)=pq$

 Let I_A be indicator r.v. of event A 

$I_A^2 = I_A, I_A^3 = I_A$

$I_AI_B = I_{A\cap B}$

VarX = npq since Cov(X_i,X_j) = 0

Ex. $X\sim HGeom(w, b, n)$  

X = X_1 + ..X_n, X_j = 1 if jth ball is white; 0 otherwise

symmetry 

$Var(X) = nVar(X_1) + 2\binom{n}{2}Cov(X_1, X_2)$ 

$Cov(X_1, X_2) = E(X_1X_2) - E(X_1)E(X_2) = \frac{w}{w+b} (\frac{w-1}{w+b-1}) - (\frac{w}{w+b})^2$  



## Lecture 22

Var of Hypergeom(w,b,n),

p = w/(w+b),  w+b = N

$Var(X) = nVar(X_1) + 2\binom{n}{2}Cov(X_1, X_2)$ 

$=\frac{N-n}{N-1} np(1-p) $

$\frac{N-n}{N-1}$  finite population correction

Extreme case n= 1, N much much larger than n

Transformations 

Therom Let X be a continous  r.v with PDG f_X, Y = g(X)

where g is different, **strictly increaing**. Then the PDF of Y is given by

$ f_Y(y) = f_X(x)dx/dy$ where y = g(x), $x =g^{-1}(y)$

and this is written in terms of y

Also $dx/dy = (dy/dx)^{-1}$  

Proof 

CDF of Y is $P(Y\le y) = P(g(X) \le y)$

$= P(X\le g^{-1}(y)) = F_X(g^{-1}(y))h = F_X(x)$

 => f_Y(y) = f_X(x)dx/dy  [chain rule]

Ex. Log normal   $Y = e^Z, Z\sim N(0,1)$

dy/dZ = e^Z= y

f_Y = 

Transfomations in multi dimension 

$\vec{Y} = g(\vec{X})$  

X = (X_1,…X_n) continuous

joint PDF of Y is f_Y(y) = f_X(x) |dx/dy | 

$|d\vec{x}/d\vec{y} |$ Jacobian  abs of value of determinant

$|dy/dx|^{-1}$    

**Convolution** (sums)

let T = X+Y, X,Y indep

discrete : $P(T=t) = \sum_{x} P(X=x)P(Y = t-x)$

continuous $f_T(t) = \int f_Xf(x)f_Y(t-x)dx$ 

since $F_Y(t) = P(T\le t) = \int P(X+Y\le t|X=x)f_X(x)dx$

$=\int F_Y(t-x)f_X(x)dx$

Idea: prove existence of objects with desired properties A using prob.

Show P(A) >0 for a random object . 

Suppose each object has a "score" . Show there is an object with "good" score. there is an object with score is at least E(X) (X score of random object)

Ex. suppose 100 people , 15 committees of 20 , each persom is on 3 committee, show there exist 2 committee with overlap $\ge$ 3    

Idea: find average overlap of 2 random committee

E(overlap) = 100 \frac{\binom{3}{2}}{\binom{15}{2}} = 20 / 7

=> there exists pair of committee with overlap of \ge 20/7 => have overlap of \ge 3

## Lecture 23

**Beta Distribution**

generalization of uniform dist.

Beta(a,b) a>0 , b>0

PDF $f(x) = c x^{a-1}(1-x)^{b-1}$ , 0<x<1

+ flexible family of continuous distribution on (0,1)

Ex, a =b = 1, uniform 

a = 1/2 = b  U shape

+ often used as prior for a parameter in (0,1)
+ "conjugate prior to Binormial "
+ connections to other distribution 

**Conjugate prior for Bin**

$X|p \sim Bin(n,p)$  $p\sim Beta(a,b)$ [prior]

Find posterior dist. $p|X$  

$f(p|X=k) = P(X=k|p) f(p)/P(X=k) $

$=\binom{n}{k}p^k(1-p)^{n-k}  c p^{a-1}(1-p)^{b-1}/P(X=k)$

$\propto p^{a+k-1}(1-p)^{b+n-k-1}$ 

=> $p|X \sim Beta(a+X, b+n-X)$ 

( P(X=k) does not depend on p)

Find$ \int_0^1 x^k(1-x)^{n-k}dx$  without using calculus

using a story (Bayse' Billiards)

n+1 billiard balls, all whit, paint one pink , throw  them at (0,1) indep

or : first throw , then paint 1 pink

let X = #balls to left of the pink one 

$P(X=k) = \int_0^1 P(X=k|p)f(p)dp$  f(p) = 1

$= \int_0^1  \binom{n}{k}p^k(1-p)^{n-k}  = 1/(n+1)$ 

 

**Stat 123: Applied ~~Quant Finance~~ Financial deriavative  on Wall Street**

$S_T $  r.v 

$g(S_T)$

$E(g(S_T))$  

**foreign exchange - FX**

Now:  one euro worth one dollar

1/2 worth more  1.25 $

1/2 worth less 0.8$

E(euro in a year) = $ 1.025 so  \$1 = euro 0.9756

E($) = euro 1.025

**TARP** 

warrants = Call option

US Govt paid $450mm to GS, 

US had right to buy \$10m GS shares for $ 125 in 10 years time 

Oct 2008 , GS shares trade  $95 

If in 10 years time , GS $150, option worth \$ 25

GS $100, 0

k = $ 125

For call option g(S_T) = max{S_T- k, 0}

prices = $\int max\{S_T- k, 0\}f(S_T)dS_T$ 



## Lecture 25

Bank-post office example

wait X time at bank to serve, $X\sim Gamma(a, \lambda)$ 

at office $Y\sim Gamma(b, \lambda)$ 

indep.

Find joint distribution of $X+Y = T, \frac{X}{X+Y} = W$  

Let \lambda = 1 to simplify notation

joint PDF $f_{T,W}(t,w) = f_{X,Y}(x,y) |d(x,y)/d(t,w)|$ 

$=\frac{1}{\Gamma(a) \Gamma(b)} x^a e^{-x}y^be^{-y}/xy  |-t|$ 

$=\frac{\Gamma(a+b)}{\Gamma(a) \Gamma(b)} w^{a-1}(1-w)^{b-1} t^{a+b}e^{-t}/t\Gamma(a+b)$  

$t^{a+b}e^{-t}/t\Gamma(a+b) = Gamma(a+b, 1)$ 

x+y = t, x/(x+y) = w => x =tw, y = t(1-w)

det |(w,t)(1-w, -t| = -t

$f_W(w) = \int f_{T,W}(t,w)dt = \frac{\Gamma(a+b)}{\Gamma(a) \Gamma(b)} w^{a-1}(1-w)^{b-1}$  

=>  know normalizing const. of  Beta(a,b)

$T\sim Gamma(a+b, 1), W\sim Beta(a,b)$  are indep. 

Find E(W), W\sim Beta(a,b)

1. LOTUS/defn 
2. $E(X/(X+Y)) = E(X)/E(X+Y)=a/(a+b)$  be careful  

Why is  E(X/(X+Y))E(X+Y) = E(X) in this special problem of Gamma-Beta?

X/(X+Y) is indep. of X+Y => uncorrelated

**Order Statistcs**

generalization of max and min problem

Let X_1, .. X_n, be i.i.d, 

The order statistics are $X_1 \le X_2 \le ..\le X_n$, where X_1 is min, X_n = max

e.g if n is even, the median is X_{(n+1)/2}  Get other "quantiles"

Difficult since dependent , Tricky in discrete case  because of "ties". 

Let$ X_1,..X_n$ be i.i.d with PDF f, CDF F, Find the CDF and PDF of $X_{(j)}$

CDF : P(X_j \le x) = P(at least j of X_i's are \le  x)

$= \sum_{k=j}^{n} \binom{n}{k} F(x)^k (1-F(x))^{n-k}$ 

PDF $f_{X(j)}(x)dx=  n \binom{n-1}{j-1}  (f(x)dx)F(x)^{j-1}(1-F(X))^{n-j} $ 

Ex. U_1, ..U_n i.i.d. Unif(0,1)

PDF of jth order statistic

$f_{Uj}(x) = n \binom{n-1}{j-1}  x^{j-1}(1-x)^{n-j}$  

=> $Uj\sim Beta(j, n-j+1)$ 

$E|U_1 - U_2| = E(max) - E(min)$

$max \sim Beta(2,1), min \sim Beta(1,2)$  



$E(X|A)  = E(X|A)P(A) + E(X|A^c)P(A^c)$

$E(X) = \sum_x xP(X=k)$ 

## Lecture 26

Two Envolope Paradox

one envolope contains twice as much as others

Argument1 E(Y) = E(X) by symmetry  —true

Argument2 E(Y) = E(Y|Y=2X)P(Y=2X) + P(Y|Y=X/2)P(Y=x/2)

=(\ne) E(2X)1/2 + E(x/2)1/2 = 5/4 E(X) — false

= E(2X|Y=2X)1/2 + E(X/2|Y=X/2)1/2 

$E(Y|Y=2X) \ne  E(2X)$ 

Let I be indicator of Y = 2X 

Then X, I are dependent



Patterns in coin filips

Repeated fair coin flips

How many flips until HT?  find E(W_{HT}) = 4

…………………………HH?  find E(W_{HH}) = 6

symmetry E(W_TT) = E(W_HH)

E(W_HT)  = E(W_TH)

$E(W_{HT}) = E(W_1) +E(W_2) = 2 + 2 = 4$, since $W_j - 1 \sim Geom(1/2)$ 

E(W_{HH}) = E(W_{HH}|1st toss H)1/2+ E(W_{HH}|1st toss T)1/2

= (2 1/2 + (2+EW_HH)1/2)1/2 + (1 + EW_HH)1/2

=> EW_HH = 6



**peter donnelly** ted.com

Y discrete

$E(Y|X=x) = \sum_y yP(Y=y|X=x)$  

Y continuous

$E(Y|X=x) = \int_{-\infty}^{\infty} yf_{Y|X}(y|x)dy $ 

X continuous

$E(Y|X=x) = \int_{-\infty}^{\infty} y\frac{f_{Y,X}(y,x)}{f_X(x)}dx$ 

Let g(x) = E(Y|X=x)

Then define E(Y|X) = g(X)

e.g. if $g(x) = x^2$, then $g(X) = X^2$ 

so E(Y|X) is a r.v  funciton of X

Ex, X,Y i.i.d Pois(\lambda)

$E(X+Y|X) = E(X|X) + E(Y|X) = X + E(Y)   = X+ \lambda$ 

E(h(X)|X) = h(X)   (X is a function itself, )

 E(X|X+Y) , Let T = X+Y, find conditional PMF 

P(X=k|T=n) = P(T=n|X=k)P(X=k)/P(T=n)

= P(Y=n-k)P(X=k)/P(T=n)

= \binom{n}{k} 1/2^n

$X|T=n \sim Bin(n, 1/2)$ 

E(X|T=n) = n/2 => E(X|T) = T/2

E(X|X+Y) = E(Y|X+Y) by symmetry since i.i.d

E(X|X+Y) + E(Y|X+Y) = E(X+Y|X+Y) = X+Y

=> E(X|T) = T/2



Iterated E (Adam'a law)

 E(E(Y|X)) = E(Y)

## Lecture 27

Ex. let $X\sim N(0, 1)$, Y =X^2

Then $E(Y|X) = E(X^2|X) = X^2 = Y$

E(X|Y) = E(X|X^2) = 0,since if observe X^2 = a , then X=+-\sqrt{a} equally likely 

Ex. stick  length 1, break off random piece, break off aonther piece

E of second piece

$X\sim Unif(0,1)  Y|X \sim Unif(0,X)$  

E(Y|X=x) = x/2, so E(Y|X) = X/2, E(E(Y|X)) = 1/4 = E(Y)

**Useful Properties**

1. $E(h(X)Y|X) = h(X)E(Y|X)$ [taking out what's known]
2. $E(Y|X) = E(Y)$, if X, Y are independent
3. $E(E(Y|X))=E(Y)$  Iterated Expectation(Adam's law)
4. $E((Y-E(Y|X))h(X)) = 0$, i.e. Y-E(Y|X) (residual) is uncorrlated with h(X)
5. $Var(Y) = E(Var(Y|X)) +Var(E(Y|X))$   EVE's law

Proof of 4. 

$E(Yh(X)) - E(E(Y|X)h(X)) = E(Yh(X)) - E(E(h(X)Y|X))$

$= E(Yh(X)) -E(Yh(X)) = 0$

Proof of 3. [discrete case]

Let E(Y|X) = g(X)

$E(g(X))= \sum_x g(x)P(X=x) = \sum_x E(Y|X=x)P(X=x)$

$= \sum_x (\sum_y yP(Y=y|X=x) )P(X=x)$

$= \sum_y \sum_x yP(Y=y, X=x) = \sum_y y P(Y=y) =E(Y)$  



Defn $Var(Y|X) = E(Y^2|X)-(E(Y|X))^2 = E((Y-E(Y|X))^2|X )$ 

Ex. Pick random city, pick random sample of n people in that city,

X =#with disease,  Q= proportion of people in the random city with disease

Find E(X), Var(X), assuming Q \sim Beta(a,b), X|Q \sim Bin(n,Q)

E(X) = E(E(X|Q)) = E(nQ) = na/(a+b)

Var(X)= E(Var(X|Q)) + Var(E(X|Q))= E(nQ(1-Q)) -n^2Var(Q)

$E(Q(1-Q))= \frac{ab}{(a+b+1)(a+b)}$

$Var(Q) = \frac{\mu(1-\mu)}{a+b+1}, \mu = a/(a+b)$ 

## Lecture 28

Ex. Store with a random # customers , N = (# customers)  Let  X~j~ be amount jth customers spend, X~j~ has mean $\mu, var, \sigma^2$  

assume N, X_1, X_2 … are indep

Find mean, var of $X = \sum_{j=1}^{N}X_j$ 

~~E(X)=N\mu~~ 

$E(X)=\sum_{n=0}^{\infty}E(X|N=n)P(N=n)=\sum_{n=0}^{\infty} \mu nP(N=n)$

$=\mu E(N)$

Adam's law $ E(X) = E(E(X|N)) = E(\mu N)=\mu E(N)$

$Var(X) = E(Var(X|N)) + Var(E(X|N))=E(N\sigma^2)+Var(\mu N)$

$=\sigma^2E(\mu) + \mu^2Var(N) $

Stat

**Inequalities** 

1. Cauchy-Schwarz: $|E(XY)| \le \sqrt{E(X^2)E(Y^2)}$  

    [if X,Y uncorrlated E(XY)=E(X)E(Y)] 

   IF X,Y have mean 0, then $|Cov(X,Y)|=|\frac{E(XY)}{(EX^2EY^2)^{1/2}}| \le 1$ 

2. Jensen's inequality: if  g is convex , then $Eg(X) \ge g(EX) $ 

   convex : $g''(x)\ge 0$  e.g. $y = x^2$ ,$y=|x|$ 

   IF h is concave , $Eh(x) \le h(EX)$  

   $EX^2 \ge (EX)^2$ 

   let X be postive , $E(1/X) \ge 1/E(X)$  $E(lnX) \le \ln E(X)$ 

3. Markov $P(|X|\ge a) \le E|X|/a$  , for any a >0

   Note that  $aI_{|X|\ge a} \le |X|$ So $aEI_{|X|\ge a} \le E|X|$  

4. Chebyshev $P(|X-\mu|> a) \le \frac{VarX}{a^2}$ 

   $P(|X-\mu|>< SD(X) ) \le \frac{1}{c^2}, c>0$  

   for $\mu = EX , a >0 $ 

   ​

Proof 2. 

$g(x) \ge a+bx$    y=a+bx is the tangent line

$g(X)\ge a+bX$ =>$ Eg(X) \ge E(a+bX) = a+bE(X) = a+b\mu = g(\mu) = g(EX)$ 

Example 3.

Ex. 100 people , Is it possible that at least 95% are younger than avg in group? - yes

Is it at 50% are older than twice avg age?  - No

Proof 4. $P(|X-\mu|>a) = P((X-\mu)^2 > a^2) \le E(X-\mu)^2/a^2=VarX/a^2$ 

## Lecture 29

let X_1,X_2… be i.i.d mean \mu, var \sigma^2,  let $\overline{X_n} = 1/n \sum_{j=1}^{n} X_j $  (sample mean)

(strong)Law of Large Numbers: $\overline{X_n} $ -> \mu as n-> \infty with probability 1

Ex. $X_j \sim Bern(p)$ , then X_1 + ..X_n /n -> p with prob. 1

(weak) LLN For c >0, $P(|\overline{X_n} - \mu| >c) -> 0$  as n-> \infty



proof weak

$P(|\overline{X_n} - \mu| >c) \le VarX_n^-/c^2 1/n^2 n \sigma^2 /c^2 = \sigma^2/nc^2 ->0$  

$X_n^- - \mu -> 0$  with prob. 1, but what does the distribution of X_n^- look like ?

Central limit Therom:  $n^{1/2}(X_n^- - \mu)/\sigma$ -> N(0,1) in distribution

Equivalently: $\frac{\sum_{j=1}^n X_j -n\mu}{\sqrt{n}\sigma}$  -> N(0,1)

Proof (assume MGF M(t) of X_j exists)

can assume \mu = 0, \sigma = 1, since consider 



Binormial Approxiamated by Normal 

Let X\sim Bin(n,p) think of X = \sum X_j , X_j \sim Bern(p), i.i.d 

$$P(a\le X \le b )=  P( \frac{a - np }{\sqrt{npq}} \le \frac{X - np }{\sqrt{npq}}  \le \frac{b - np }{\sqrt{npq}}  )​$$ 

$\approx \Phi (\frac{b - np }{\sqrt{npq}}) - \Phi(\frac{a - np }{\sqrt{npq}})$ 

contrast with Pois approx

Pois n large p small , \lambda = np

Normal n large , p close to 1/2



$P(X=a) = P(a-1/2 \le a \le 1+1/2)$   a-integer

## Lecture 30

**Chi-square**

$\chi^2(n)$   (Chi-square) 

Let $V=Z_1^2 + Z_2^2 +..+Z_N^2$ i.i.d N(0,1)

then $V\sim \chi^2(n)$  

Fact $\chi^2(1)$ is $Gamma(1/2, 1/2)$  

So $\chi^2(n)$ is $Gamma(n/2, 1/2)$  

**Student - t** (Gosset, 1908)

Let $T = \frac{Z}{\sqrt{V/n}}$ , with $Z\sim N(0,1) ,V\sim \chi^2(n)$ indep.

Then $T\sim t_n$  

Properties:

1. symmetric, i.e $-T \sim t_n$  

2. $n=1 \Rightarrow Cauchy$ , mean doesn't exist

3. $n\ge 2 \Rightarrow E(T) = E(Z)E( \frac{1}{\sqrt{V/n}}) = 0$  

4. Heavier-tailed than Normal

5. For n large,t_n looks very much like N(0,1)

   distribution of t_n goes to N(0,1) as $n\to \infty$  

$E(Z^2) = 1, E(Z^4) = 3, E(Z^5)=3\times 5 ..$  used MGF

Another way:

$E(Z^{2n})=E((Z^2)^n), Z^2 = \chi^2(1) \sim Gamma(1/2, 1/2)$

Let $T_n = \frac{Z}{\sqrt{V/n}}$ , with $Z\sim N(0,1) ,$ indep.

$ V_n=Z_1^2 + Z_2^2 +..+Z_N^2$

Then $V_n/n \to 1$ with prob. 2 by LLN since $EZ_1^2 = 1$

so $T_n \to Z$  with prob. 1 So $t_n$ converges to N(0,1)

**Multivariate Normal** (MVN)

Defn Random vector $(X_1, X_2,…X_k) = \vec{X}$ is Multivariate Normal if every linear combination $t_1X_1 + t_2X_2 + ..t_kX_k$  is Normal

Ex Let Z,W be i.i.d N(0,1) Then (Z+2W, 3Z+5W) is MVN , 

since s(Z+2W) + t(3Z+5W) = (s+3t)Z+(2s+5t)W is Normal 

Non-Ex.

 Z\sim N(0,1), let S be random indep. of Z, then Z, SZ are marginally N(0,1), But (Z,SZ) is not Normal. look at Z+SZ 

Let $EX_j = \mu_j$, MGF of $\vec{X}$ (MVN) is $E(e^{\vec{t}'\vec{X}})  = E(e^{t_1X_1+t_2X_2+…+t_kX_k})$

$= exp((t_1\mu_1+..+t_k\mu_k)+1/2\times Var(t_1X_1+t_2X_2+…+t_kX_k))$  

Theorm: within MVN, uncorrlated implies indep

$\vec{X} = \binom{\vec{X_1}}{\vec{X_2}}$ MVN, id every component of $\vec{X_1}$ is uncorrlated with every component of $\vec{X_2}$,then $X_1$ is indep. of $X_2$ 

Ex. Let X,Y be i.i.d N(0,1) Then (X+Y, X-Y) is MVN, 

uncorr: $Cov(X+Y, X-Y) = Var(X) + Cov(X,Y)-Cov(X,Y)-Var(Y) = 0$ 

So X+Y, X-Y are indep.

## Lecture 31



