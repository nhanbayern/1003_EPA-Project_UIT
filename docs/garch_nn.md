## From GARCH to Neural Network for Volatility Forecast

## Pengfei Zhao*^1 , Haoren Zhu*^2 , Wilfred Siu Hung NG^2 , Dik Lun Lee^2

(^1) Guangdong Provincial Key Laboratory of Interdisciplinary Research and Application for Data Science, BNU-HKBU United
International College,
(^2) Hong Kong University of Science and Technology,
ericpfzhao@uic.edu.cn,{hzhual, wilfred, dlee}@cse.ust.hk,
Abstract
Volatility, as a measure of uncertainty, plays a crucial role in
numerous financial activities such as risk management. The
Econometrics and Machine Learning communities have de-
veloped two distinct approaches for financial volatility fore-
casting: the stochastic approach and the neural network (NN)
approach. Despite their individual strengths, these method-
ologies have conventionally evolved in separate research tra-
jectories with little interaction between them. This study en-
deavors to bridge this gap by establishing an equivalence re-
lationship between models of the GARCH family and their
corresponding NN counterparts. With the equivalence rela-
tionship established, we introduce an innovative approach,
named GARCH-NN, for constructing NN-based volatility
models. It obtains the NN counterparts of GARCH models
and integrates them as components into an established NN
architecture, thereby seamlessly infusing volatility stylized
facts (SFs) inherent in the GARCH models into the neural
network. We develop the GARCH-LSTM model to show-
case the power of the GARCH-NN approach. Experiment re-
sults validate that amalgamating the NN counterparts of the
GARCH family models into established NN models leads
to enhanced outcomes compared to employing the stochas-
tic and NN models in isolation.

## Introduction

```
In empirical finance, the volatility of asset returns is a mea-
sure of risk, which helps investors decide if the returns of
assets justify the risks. Further, volatility is used by finan-
cial institutions to assess their risks using value-at-risk (VaR)
models. The ability to forecast the volatility or variance of
asset returns can help investors to anticipate risks and to re-
duce loss, which is important for risk management. Finan-
cial forecasting can be categorized into stochastic and ma-
chine learning (ML) approaches. Existing econometrics so-
lutions fall into the stochastic category. For example, the fa-
mousgeneralized autoregressive conditional heteroscedas-
ticity(GARCH) family models for volatility modeling have
been widely used in risk modeling for decades. Stochas-
tic econometric time series models have the advantage that
they can be theoretically described based on statistical logic.
However, they rely on many assumptions (e.g., explanatory
*These authors contributed equally.
Copyright © 2024, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
```
```
variables must be stationary), which may not align with the
highly dynamic, nonlinear, and complex market reality.
Recently, ML approaches have been increasingly used in
volatility forecast (Ge et al. 2022). However, ML approaches
are mostly uninterpretable and lack support from financial
domain knowledge. While ML models have the power to
approximate nonlinear functions and fewer restrictions, they
have several limitations: (1) The complexity of NN mod-
els poses challenges in explaining the model’s outputs, thus
reducing confidence in the model’s reliability. The issue is
magnified in the financial sector where risks and responsi-
bilities are at stake, compelling practitioners to understand
how forecasts and decisions are made to build up trust in
the model. (2) There are a large number of models that
fit the training data well, but few generalize well (Giles,
Lawrence, and Tsoi 2001). The challenge mainly comes
from the non-stationarity and noisiness of real-world finan-
cial time series data. A small training timeframe may in-
duce overfitting in NN models due to the presence of noise.
Conversely, an excessively large timeframe can introduce
inconsistency between the temporal relationship of inputs,
confusing NN models during the learning process due to
data non-stationarity. Recent work (Zeng et al. 2022) shows
that simple one-layer linear models outperform sophisti-
cated Transformer-based models on long-term time series
forecasting. (3) Well-established NN models are often gen-
eral ML solutions and the network structure is not designed
to capture the characteristics of the volatility time series.
Thus, statistical models like GARCH are still the dominant
volatility forecast models until now due to their simplicity,
high interpretability, and satisfactory performance.
Stochastic and NN solutions have complementary
strengths, but existing studies treat them as separate research
directions without much interaction in between. This pa-
per aims to develop an approach that can exploit the well-
established foundation of the GARCH family of volatil-
ity models in NN-based volatility models to improve pre-
diction accuracy as well as enhance the interpretability of
the NN models. In particular, econometricians have discov-
ered the fact that volatility, unlike most market variables
that remain largely unpredictable, has specific characteris-
tics named “stylized facts” (SFs) (Masset 2011) (also re-
ferred to as “volatility characteristics” in the literature) that
can increase the accuracy of volatility forecast. Examples of
```
# arXiv:2402.06642v1 [q-fin.ST] 29 Jan 2024


stylized facts are: the change of volatility tends to persist
(volatility clustering), to be higher in declining markets than
in rising markets (asymmetric effect), and has a long-lasting
impact on its subsequent evolution (long memory). SFs are
intuitive and reflect human behaviors in response to risks. As
shown later in our experiments, NN models that incorporate
SFs have better accuracy compared to SOTA neural models.
Our approach begins by studying the translation of es-
tablished GARCH family econometric volatility models
into their NN counterparts. The NN representations of the
GARCH family that embed SFs then serve as building
blocks plugged into the broader NN framework, aligning
stochastic and NN approaches harmoniously. We call this
approach the GARCH-NN approach. A major benefit is that
the mathematical structures and statistical properties of the
GARCH family models have been well studied with rigor by
econometric researchers. The NN counterpart of the well-
understood GARCH model can serve as the blueprint for
building the final NN volatility model that can be understood
and trusted by practitioners in the same way they trust the
original stochastic models (Tjoa and Guan 2021), helping to
enhance NN volatility model’s interpretability.
Drawing from this perspective, we propose the GARCH-
LSTM model as a motivative example for developing NN-
based volatility models from stochastic volatility models.
GARCH-LSTM seamlessly integrates GARCH’s NN coun-
terpart into the LSTM architecture. This design empowers
the NN model not only to capture the stylized facts from
GARCH but also to leverage LSTM’s ability for long-short
memory. Experiment results reveal that the amalgamation
of the GARCH family (statistical) and LSTM (ML) models
improves forecasting accuracy compared to their individual
use. The contributions of our paper are listed as follows:

- According to the authors’ best knowledge, we are the
    first to study the equivalence relation between the clas-
    sic GARCH family models in the econometrics field and
    NN in the ML field. This observation helps to bridge the
    gap between stochastic and ML approaches in financial
    volatility forecasting.
- Our approach constructs NN-based volatility models that
    integrate stochastic volatility models as building blocks.
    In this way, the stylized facts are directly infused into the
    NN framework, unlike traditional methods that simply
    take stochastic model outputs or parameters as input of
    the NN model. We develop the GARCH-LSTM model to
    demonstrate the superiority of our approach.
- We utilize five globally traded equities time series
    datasets and employ the GARCH-LSTM framework to
    compute the Value at Risk (VaR). Experiment results val-
    idate the existence of the GARCH-NN equivalence rela-
    tion and combining the fundamental statistical (GARCH
    family models) and ML (LSTM) models yields improved
    results compared to employing each model in isolation.

## Related Work

### Stochastic Volatility Models

The GARCH model (Bollerslev 1986) is a widely used sta-
tistical model that captures volatility clustering and time-

```
varying volatility in financial time series data. GARCH(1,1)
is widely used in practice due to its simplicity and supe-
rior performance (Bollerslev 1987). Many extensions have
been developed after the pioneering work of GARCH. GJR-
GARCH (Glosten, Jagannathan, and Runkle 1993) aims
to capture the asymmetry in the response of volatility,
whereby negative shocks have a stronger impact compared
to positive shocks. Early GARCH models were constrained
by a short memory due to the exponential decay of the
volatility weights. Notably, FI-GARCH (Baillie, Bollerslev,
and Mikkelsen 1996) and its extensions (Davidson 2004;
KIlIc ̧ 2011) incorporate hyperbolic decaying coefficients,
enabling these models to “memorize” long historical terms.
MM-GARCH (Li, Li, and Li 2013) combines elements of
both short-memory and long memory GARCH models.
```
### Neural Network Volatility Forecasting Models

```
Although the econometric time series models are advanta-
geous in that they can be theoretically described based on
statistical logic, they rely on various assumptions that do
not align with the reality of highly dynamic and complex
financial markets. Over the past few years, significant ad-
vancements have been made in general-purpose deep learn-
ing (DL) models for time series forecasting, leveraging at-
tentional mechanisms and transformer structures (Vaswani
et al. 2017; Zhou et al. 2021; Kitaev, Kaiser, and Levskaya
2020; Salinas et al. 2020; Oreshkin et al. 2019; Sen, Yu, and
Dhillon 2019; Lai et al. 2018; Wu et al. 2021). Recent re-
search has reworked linear models in time series forecasting
and showcased their superiority in specific contexts (Zeng
et al. 2022; Chen et al. 2023). However, there exists evidence
that DL struggles to outperform classical stochastic time se-
ries forecasting approaches (Makridakis, Spiliotis, and Assi-
makopoulos 2018; Elsayed et al. 2021), possibly due to that
general-purpose NN models are not originally designed for
the volatility forecast task to include unique stylized facts.
Attempts have been made to incorporate stylized facts
into the ML model. Most hybrid solutions take the outputs or
parameters of stochastic models as features input to differ-
ent types of NNs, like Multi-layer perceptron (Khan, Hasan-
abadi, and Mayorga 2017; Pyo and Lee 2018; Kristjanpoller
and Minutolo 2016), LSTM (Kristjanpoller, Fadic, and Min-
utolo 2014; Liu and So 2020; Rahimikia and Poon 2020;
Kim and Won 2018), Transformer (Ramos-Perez, Alonso- ́
Gonzalez, and N ́ u ́nez-Vel ̃ azquez 2021), and attention neural ́
network (Lin and Sun 2021; Zheng et al. 2019). The ensem-
ble approach combines the outputs from GARCH and NN
(Kakade, Jain, and Mishra 2022; Hu, Ni, and Wen 2020).
(Ge et al. 2022) systematically reviewed NN-based volatility
forecasting. However, simply staggering two different mod-
els in a pipeline does not guarantee the interpretability or
effectiveness of the approach. Our paper distinguishes it-
self from existing NN-based volatility forecasting models
in that weencode SFs into the NN structure. The end-to-
end GARCH-NN framework accelerates the model training
and makes online forecasts feasible. Compared to sophisti-
cated DL time series models, we intentionally choose basic
NN models to build the fundamental equivalence relation
between NNs and their stochastic GARCH counterparts.
```

## Preliminaries and Problem Definition

The GARCH family has become the most popular way of
parameterizing the dependence in volatility time series. The
GARCH(1,1) model can be expressed in Equation 1, where
rt=logppt−t 1 denotes the log return att,Dstands for the dis-

tribution (typically assumed to be a normal or a leptokurtic
one),{εt}may be observed directly, or it may be a residual
sequence of an econometric model,ψt− 1 denotes the histor-
ical information,γis a scalar, andθ(d) =αβd−^1 represents
the contribution ofε^2 t−din forecastingσ^2 t, which is an expo-
nential time decayed function which dies out quickly with
the increment of time lagddue to the autoregressive term,
leading to a short memory. 0 < α+β < 1 guarantees the
stationarity of the GARCH process.

```
rt=μt+εt, εt|ψt− 1 ∼D(0,σ^2 t)
```
```
σ^2 t=ω+α∗ε^2 t− 1 +β∗σ^2 t− 1 =γ+
```
#### X∞

```
d=
```
```
θdε^2 t−d
```
#### (1)

Given an input univariate time series withltime steps,
Etl ={εt,εt− 1 ,...,εt−l+1},l≤ L, whereLdenotes the
entire time series length. According to the efficient market
hypothesis, in practice, we treatμt= 0and thenεt=rtin
Equation 1. The forecasting problem is defined below:

```
σˆ^2 t+h=F(Etl|ΘF) (2)
```
whereˆσ^2 t+hdenotes the forecast future volatility att+h,h
denotes the forecast horizon,Fdenotes the NN forecasting
model, andΘFrefers to the model parameters. The goal is to
design a highly interpretable model, meaningFeither, has a
well-studied NN structure whose property and performance
are comprehensively studied and recognized or has equiva-
lent mathematical counterparts having clear econometrics/s-
tatistical meaning, or both.

## Methodology

The GARCH-NN approach of creating interpretable NN
volatility models involves two key steps. Firstly, we establish
an equivalence between GARCH models and their NN coun-
terparts. Secondly, we seamlessly integrate the NN counter-
part of GARCH into established NN blocks (such as LSTM),
thereby ensuring the preservation of volatility stylized facts
captured by GARCH within the NN framework.

### Equivalence Relation Between GARCH and NN

In this paper, the GARCH-NN equivalence is defined as both
models sharingidentical inputs, model structure, model pa-
rameters, loss function, and training processes.

Equivalence of Model Structure and Parameters Due
to the page limit, we choose the three fundamental GARCH
family models, namely, GARCH(1,1), GJR-GARCH, and
FI-GARCH, corresponding to the volatility clustering,
asymmetry, and long memory volatility stylized facts re-
spectively. The NN counterparts of other GARCH family
models can be established similarly.

```
(a) Basic RNN Structure (b) GARCH(1,1)
```
```
(c) GJR-GARCH (d) FI-GARCH
```
```
Figure 1: Equivalence between GARCH models and their
NN counterparts.
```
```
We start with GARCH(1,1), the simplest version of
GARCH models. Figure 1b illustrates the recursive structure
of the GARCH(1,1) defined in Equation 1, where the model
input isσ^2 t− 1 andε^2 t− 1 , and model outputσ^2 tis treated as the
next GARCH cell’s input. Intuitively, we can observe that
it shares an identical recurrent structure as the RNN in Fig-
ure 1a. Specifically, if we remove RNN’s output layer and
tanhactivation, and at the same time constrain both the in-
put and hidden state to a scalar, the two models’ structures
are identical. Thus, theoretically, the GARCH(1,1) model
is a special case of the RNN model with the truncation of
the output layer and activation function. The RNN cell con-
tains the parameter listΘgarch 11 = (ω,α,β)and receives
the observation listXgarch 11 = (1,ε^2 t− 1 ,σ^2 t− 1 )and the out-
putσ^2 tis the linear combinationΘgarch 11 ·Xgarch 11. Fig-
ure 1c illustrates the RNN equivalence of the GJR-GARCH
model. Compared to the GARCH(1,1) equivalence, the in-
put to the RNN cell isεtinstead ofε^2 tsince the sign in-
formation ofεtis required. The RNN cell contains the pa-
rameter listΘgjr = (ω,α,λ,β)and the observation list
Xgjr= (1,ε^2 t,I(εt)∗ε^2 t,σt^2 − 1 ), whereIis the sign function
and the outputσ^2 t= Θgjr·Xgjr.
In FI-GARCH(1,d, 1), the fractionaldcontrols the degree
of long-memory behavior, and the parameter listΘfigarch=
(ω,β,φ)defines a decaying scheme for lags ofε^2 t. The co-
efficient ofε^2 thas the series expansion form as follows:
```
#### [1−

```
1 −φB
1 −βB
```
```
(1−B)d]ε^2 t= (
```
#### X∞

```
k=
```
```
λkBk)ε^2 t≈
```
#### TX− 1

```
k=
```
```
λkε^2 t−k
```
```
(3)
wherekdenotes the lagging step,Bdenotes the backshift
operator (e.g.Bkε^2 t = ε^2 t−k), and eachλkis a function
of(β,φ,d). In practice, a finite number of terms is ob-
tained in the series expansion by using a truncation sizeT.
LetΛ = (λ 0 ,...,λT− 1 )be the weights computed based on
(β,φ,d). Sinceλkis independent witht, we can viewΛas
a sliding window over the{ε^2 t}series and this is similar to
```

applying a 1-d convolutional operation to{ε^2 t}. Thus, the
long-memory nature of the FI-GARCH model can be rep-
resented by a basic Convolutional Neural Network (CNN)
structure, as illustrated in Figure 1d. The kernel size is equiv-
alent to the truncation sizeTand the convolutional weights
areΛ. The CNN cell slides over the{ε^2 t}series to perceive
Xfigarch={ε^2 t−T+1,...ε^2 t}andσ^2 t= Λ·Xfigarch.

Equivalent Loss Function While existing NN-based
volatility forecast models commonly employ mean squared
error (MSE) or mean absolute error (MAE) loss functions
between predicted volatilityσˆtand true volatilityσt, this can
present practical challenges to volatility forecast due to the
inherent statistical nature of volatility and its various forms
such as historical, implied, and realized volatility (Ge et al.
2022). To establish the GARCH-NN equivalence, we opt for
the maximum likelihood approach as the loss function, a
choice aligned with stochastic volatility models, as shown
in Equation 4, whereNsignifies the sequence length.

```
arg max
Θ
```
```
ΠNt=1L(εt; Θ) = arg min
Θ
```
#### XN

```
t=
```
```
−logL(εt; Θ) (4)
```
Assumeεtin Equation 1 satisfiesεt|ψt− 1 ∼N(0,σˆ^2 t; Θ),
then the negative log-likelihood is denoted in Equation 5,
termed asN-loss.

```
−logL(εt; Θ) =
```
```
logˆσt(Θ)^2
2
```
#### +

```
ε^2 t
2ˆσt(Θ)^2
```
#### (5)

We also adopt Student’stdensity function in our experi-
ments since studies (Liu and So 2020) have shown that it is
suitable for volatility forecast. Then,εtsatisfiesεt|ψt− 1 ∼
S(0,σˆ^2 t,v; Θ)with a meanμ= 0, varianceˆσt(Θ)^2 and de-
gree of freedomv. The negative log-likelihood is then de-
noted as Equation 6, termed asT-loss.

```
−logL(εt; Θ) =
```
```
log ˆσt(Θ)^2
2
```
#### +

```
v+ 1
2
```
```
log
```
#### 

#### 1 +

```
ε^2 t
(v−2)ˆσt(Θ)^2
```
####  (6)

The equivalence relation between the GARCH models
and their NN counterparts can be established formally using
identical model structures, parameters, and loss functions,
while also feeding the same input and applying the same
training settings, such as the SLSQP optimizer that is com-
monly used in sequential stochastic models.

### GARCH-LSTM Model

LSTM is the classic sequential ML model whose capabil-
ity to balance long and short historical information (mem-
ory) has been well studied and verified in various prediction
tasks. Our goal is to design a model, named GARCH-LSTM,
empowered with both the traditional GARCH family’s ca-
pability of modeling the SFs and LSTM’s ability to balance
the long and short memory of volatility history. Equation 7
denotes the GARCH-LSTM model. Since there are many
variations of the GARCH family models, we adopt a loosely

```
coupled design that allows different GARCH models to eas-
ily plug into the LSTM framework. Specifically, in contrast
to the conventional interpretation of LSTM’s output gateot
as regulating the information selection fromct, we reinter-
pretotas the GARCH output andctas the controller infus-
ing LSTM’s long and short-term memory effect.Kgarchde-
notes the GARCH kernel function which can be flexibly re-
placed by different GARCH models’ NN counterparts (e.g.
GARCH(1,1), GJR-GARCH, FI-GARCH) introduced in the
previous section. Notably,εt− 1 is the input to the model in-
stead ofεtwhich is originally used in LSTM sinceεtcannot
be sampled before knowingσt, as shown in Equation 1. We
modify the output structure in the last line of Equation 7 by
multiplying the output of the GARCH kernel function with
(1 +w∗tanh(ct)). Influence from LSTM is ignored when
w = 0and the model is shrunk to GARCH’s NN coun-
terpart. With the increment ofw, the LSTM module would
have a greater impact on the final output by either magnify-
ingot(ct> 0 ) or shrinkingot(ct< 0 ). GARCH-LSTM
seamlessly integrates the GARCH family’s NN counterparts
encapsulating SFs into the LSTM framework.
```
```
ft=σg(Wf∗εt− 1 +Uf∗σ^2 t− 1 +bf)
it=σg(Wi∗εt− 1 +Ui∗σ^2 t− 1 +bi)
ot=Kgarch(εt− 1 ,σ^2 t− 1 ; Θ)
c ̃t=σc(Wc∗εt− 1 +Uc∗σ^2 t− 1 +bc)
ct=ft⊙ct− 1 +it⊙c ̃t
σt^2 =ot⊙(1 +w∗tanh(ct))
```
#### (7)

## Experiment

### Experiment Settings

```
Datasets We select five asset types covering stock indexes,
exchange rates, and gold prices^1 , which are widely traded
equities by investors from all over the world. Following
the common practice in the volatility forecasting literature
(Bucci 2020; Bucci et al. 2017; Andersen et al. 2003), we
obtain the return series by computing the logarithm differ-
ence based on the daily close price seriesrt =logptp−t 1 ,
and generate the realized volatility based on thek-day aver-
age asσt =
```
```
qP
k− 1
i=0ε
```
```
2
t−i: Here we set the window size
k = 5. The dataset is comprised of records in the form
of[(εt−k+1,...,εt),σ^2 t+h]. To avoid numeric underflow, we
multiply theεtandσt^2 +hby a factor 100. Table 1 summa-
rizes the statistics of different datasets. ADF and p-value
columns refer to the Augmented Dickey-Fuller test statis-
tic and p-value. The near-zero p-value rejects the null hy-
pothesis that a unit root is present in the time series data,
indicating the stationarity of the time series. We also con-
ducted the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
for each dataset, and the results failed to reject the null hy-
pothesis that the time series is stationary. We split the com-
plete dataset into training, validation, and testing parts and
the split ratio is roughly 8:1:1. We train and tune algorithms
```
(^1) Data were downloaded from https://finance.yahoo.com/


```
Dataset Length Mean Sd ADF P-value
S&P 500 2514 0.0414 1.078 -15.99 6.72e-
DJI 2515 0.0314 1.93 -15.93 7.76e-
NASDAQ 2516 0.0574 1.261 -13.66 1.56e-
EUR-USD 2603 -0.0109 0.497 -22.08 0.
Gold 2514 -0.0012 0.984 -51.95 0.
```
```
Table 1: Summary of dataset statistics.
```
```
Category
```
```
Method GARCH NN Counterparts
```
```
GARCH(1, 1)
```
```
ω 0.00359(0.00659) 0.00429(0.00659)
α 0.00540(0.00596) 0.00543(0.00563)
β 0.00923(0.0131) 0.0116(0.0161)
```
```
GJR-GARCH
```
```
ω 0.00068(0.00073) 0.00081(0.00082)
α 0.00073(0.00087) 0.00124(0.00119)
β 0.00364(0.00291) 0.00415(0.00370)
λ 0.00451(0.00748) 0.00363(0.00588)
```
```
FI-GARCH
```
```
ω 0.164(0.367) 0.0827(0.119)
β 0.0211(0.0201) 0.0363(0.0441)
φ 0.0189(0.0201) 0.0195(0.0375)
d 0.0185(0.0282) 0.0249(0.0266)
```
Table 2: Parameter estimation comparison between GARCH
models and their NN counterparts using simulation data.

using the training and validation sets, respectively, and eval-
uate the model performance based on the testing set.

Metrics Mean Squared Error (MSE) and Mean Absolute
Error (MAE) between the ground truth volatility and the pre-
dicted volatility are used for performance evaluation.

### Validation of GARCH-NN Relation

To validate the GARCH-NN equivalence relation in prac-
tice, we compare the training process, learned parameters,
and model outputs (forecasting results) of GARCH models
and their NN counterparts.

Training Process We implement the NN counterparts
in PyTorch. The vanilla GARCH models use the SLSQP
optimizer. Theoretically, the equivalence relation requires
GARCH and its NN counterpart to have the same optimizer,
while in practice, for better adaptability to NN architectures,
we select the ADAM optimizer and use a dynamic learning
rate scheduler which reduces the rate by a factor of 2 if the
validation performance stagnates. We use an early-stopping
mechanism that terminates the training process if the model
performance on the validation dataset does not improve for
more than 20 epochs. Both the vanilla GARCH models and
their NN counterparts are trained by rolling over the train-
ing part of the time series and generating the out-of-sample
predictions based on the same testing dataset.

Comparison of Learned Parameters with Simulation
Data The close parameter estimation results of the
stochastic GARCH models and their NN counterparts would
indicate the validity of the equivalence relation. Since we
do not have knowledge about the ground truth generation
process of the real-world time series, we evaluate the pa-

```
rameter estimation results using simulation data. For each
equivalence scenario, we generate 8 groups of simulated re-
turn{εt}series, each generated from the pre-fixed ground
truth parameters ranging from (0.1,0.9) according to the cor-
responding GARCH process. The stochastic GARCH mod-
els and their NN counterparts are trained on the simulation
data by the standard ARCH package (Sheppard et al. 2022)
and PyTorch, respectively, and MSE is calculated between
the estimated parameter values and the ground truth param-
eter values. Table 2 displays the average MSE and the stan-
dard deviation for each estimated parameter. From the exper-
imental results, we can observe that both statistical and ML
parameter estimation have low MSE compared to the orig-
inal scale of parameters and this demonstrates the validity
of the GARCH-NN equivalence relation. The experimental
results also point out a way to automate the parameter esti-
mation process of the GARCH family models, since as long
as the NN counterpart is found, parameter estimation can be
solved with backpropagation which saves great efforts for
econometric models to develop explicit case-by-case param-
eter estimation algorithms.
```
```
Comparison of Model Outputs We further compare the
forecasting performance of the stochastic GARCH mod-
els and their NN counterparts using real-world time series
data. If the forecasting performance is similar for both types
of models, it would support the equivalence relation. Both
vanilla GARCH models and their NN counterparts learn the
model parameters based on the training and validation parts
of the real-world datasets. Prediction is done in the test-
ing dataset with horizonh= 1. Table 3 shows the experi-
ment results where “vanilla” denotes the stochastic volatility
model using ARCH as the forecast package. Results show
that the forecast performance of the stochastic models and
their NN counterparts are close, which demonstrates the va-
lidity of the GARCN-NN equivalence relation.
```
### Model Evaluation and Analysis

```
In this section, we evaluate and analyze the performance of
the proposed framework together with the baseline models.
```
```
Impact of GARCH Loss Function Above we introduce
the GARCH loss function N-loss and T-loss based on max-
imizing the likelihood of{εt}in Equation 1. In this subsec-
tion, we compare their performances with MSE loss which
is widely used in existing NN-based volatility forecast mod-
els. The experiment is conducted based on the classical RNN
and LSTM models using the loss functions of MSE, N-loss,
and T-loss (with different degrees of freedom). We com-
pare their performances in forecasting the volatility values
with forecast horizonh = 1(one day later) in the test-
ing dataset. Table 4 shows the performance of different loss
functions. We can see that N-loss and T-loss achieve signif-
icant performance enhancement compared with MSE loss
in most datasets, demonstrating the superiority of the max-
imum likelihood-based loss function in the volatility fore-
casting task. The optimal performance is highlighted and we
can see the T-loss with freedom (v=5) has the best perfor-
mance in most cases, thus we select it as the loss function
for training GARCH-LSTM in the following subsection.
```

```
Method
```
```
Dataset S&P 500 DJI NASDAQ EUR-USD Gold
MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE
```
```
GARCH(1,1) VanillaNN Counterpart 0.205 0.0760.221 0.086 0.175 0.0560.191 0.065 0.310 0.1720.313 0.173 0.116 0.0240.116 0.027 0.274 0.1230.267 0.
```
```
GJR-GARCH VanillaNN Counterpart 0.207 0.0740.203 0.071 0.190 0.0600.191 0.061 0.296 0.1570.283 0.143 0.114 0.0230.111 0.021 0.278 0.1260.271 0.
```
```
FI-GARCH VanillaNN Counterpart 0.213 0.0760.227 0.091 0.183 0.0570.194 0.058 0.305 0.1700.305 0.227 0.128 0.0290.131 0.057 0.279 0.1250.298 0.
```
```
Table 3: Compare forecast performance with real-world dataset between stochastic models and their NN counterparts.
```
```
Dataset
```
```
Method RNNMSE lossLSTM RNNN-LossLSTM RNNT-loss (v=3)LSTM RNNT-loss (v=5)LSTM
MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE
S&P 500 0.765 1.001 0.725 0.909 0.247 0.100 0.272 0.124 0.598 0.448 0.505 0.443 0.200 0.068 0.216 0.
DJI 0.398 0.263 0.682 0.753 0.227 0.096 0.235 0.101 0.184 0.050 0.389 0.258 0.198 0.062 0.182 0.
NASDAQ 0.748 0.952 1.017 1.723 0.461 0.369 0.442 0.385 0.288 0.128 0.805 1.142 0.432 0.336 0.354 0.
EUR-USD 0.164 0.043 0.165 0.043 0.310 0.121 0.132 0.027 0.168 0.040 0.190 0.051 0.156 0.038 0.150 0.
Gold 0.321 0.175 0.609 0.558 0.298 0.143 0.325 0.186 0.347 0.188 0.302 0.159 0.298 0.152 0.293 0.
```
```
Table 4: Impact of different loss functions. The bold number indicates the best performance.
```
Overall Comparison To provide a comprehensive evalua-
tion of the proposed framework, we compare the forecasting
performance of the GARCH-LSTM model to both the state-
of-the-art deep learning time series forecasting algorithms
and the typical stochastic models with respect to different
forecast horizons. Specifically, we include four transformer-
based models, namely,Autoformer(Wu et al. 2021),In-
former(Zhou et al. 2021),Reformer(Kitaev, Kaiser, and
Levskaya 2020), and the original Transformer(Vaswani
et al. 2017), in the overall comparison. For these models, we
input the returns series to the encoder and input the volatil-
ity series to the decoder to generate future volatility. We set
the input length and label length to 126 and use grid search
to obtain the optimal hyperparameters. For GARCH-LSTM,
we tune the initial learning rate to 1e-2 from the range [3e-2,
3e-4] and use GJR-GARCH as the kernel function. To avoid
potential information leaks, we ensure that all approaches
are evaluated based on the same testing dataset and no test-
ing data sample is used in the training stage.
Table 5 shows the overall comparison results concerning
different horizon values. Here, ”1D”, ”3D”, ”1W”, ”2W”,
and ”1M” refer, respectively, to the horizon value of one
trading day (h= 1), three days (h= 3), one week (h= 5),
two weeks (h= 10), and one month (h= 21). We report
the averaging results based on 10 runs with different ran-
dom seeds. We can observe that GARCH-LSTM achieves
the best MAE and MSE in most dataset and horizon settings.
Specifically, GARCH-LSTM has dominating performance
in S&P 500, NASDAQ, and Gold datasets, with around3%
MAE and10%MSE improvement on average against the
second-best approach. Most transformer-based methods do
not perform well and this is aligned with the finding from
(Zeng et al. 2022). The three stochastic models have steady
performance although their comparison results may vary a
bit due to the different characters of the five datasets. In gen-
eral, most methods’ forecasting capabilities decayed as hori-
zons become larger and this is natural due to the uncertain

```
and dynamic long-horizon future. In terms of stability and
robustness, the proposed method generally exhibits smaller
variances ( ̃1e-4) compared to deep learning models ( ̃1e-3).
```
### Application Study

```
In the real-world financial industry,Value at Risk(VaR) is
widely used in risk management. It seeks to measure mar-
ket risks in terms of asset price volatility, which synthesizes
the greatest (or worst) loss expected from a portfolio, within
determined time periods and confidence intervals. Formally,
VaR is defined for a long position in an assetSover a time
horizonj, with probabilityp( 0 < p < 1 ):
```
```
p=P(∆Pj≤V aR) =Fj(V aR) (8)
where∆Pjrepresents the gain or loss amount of position
P, given by∆Pj=|Pt+j−Pt|andFj(.)is the accumu-
lated distribution function of the random variable∆Pj. For
example, if a portfolio of stocks has a one-day 5% VaR of
$1 million, that means that there is a 0.95 probability that
the portfolio position changes∆Pjwill fall in value by less
than $1 million over a one-day period if there is no trading.
To calculate the VaR it is necessary to have an estimate of the
volatility of the asset’s log returns for the analysis horizon.
The successful forecast of future volatility helps establish
a reliable VaR, and in turn, the count of the number of times
violating VaR would also help examine the volatility fore-
cast performance (Galdi and Pereira 2007). Thus, besides
the MSE and MAE evaluation in Table 5, in this study, we
count the violations of the VaR limits given by the number of
excesses outside the confidence interval. The smaller viola-
tion rate indicates a better volatility forecast performance.
The upper (lower) violation rate refers to the percentage
that returns exceed the upper (lower) VaR limits over the
total length of returns. In this study, we evaluate three differ-
ent approaches, GARCH(1,1), GJR-GARCH, and GARCH-
LSTM, that have top performances in Table 5 based on the
```

```
Methods GARCH-LSTM Autoformer Informer Reformer Transformer GARCH(1, 1) GJR-GARCH FI-GARCH
Metrics MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE MAE MSE
```
```
S&P 500
```
```
1D 0.205 0.070 0.636 0.546 0.494 0.387 0.489 0.381 0.494 0.386 0.205 0.076 0.207 0.074 0.213 0.
3D 0.250 0.115 0.576 0.483 0.497 0.391 0.497 0.395 0.497 0.390 0.276 0.136 0.258 0.121 0.253 0.
1W 0.332 0.195 0.606 0.542 0.497 0.391 0.497 0.394 0.497 0.390 0.358 0.225 0.346 0.212 0.336 0.
2W 0.385 0.256 0.571 0.488 0.491 0.386 0.490 0.387 0.491 0.385 0.413 0.288 0.410 0.293 0.399 0.
1M 0.430 0.304 0.567 0.523 0.485 0.381 0.485 0.386 0.485 0.380 0.441 0.326 0.439 0.316 0.430 0.
```
```
DJI
```
```
1D 0.170 0.056 0.478 0.389 0.394 0.244 0.393 0.248 0.394 0.246 0.175 0.056 0.190 0.060 0.183 0.
3D 0.269 0.107 0.441 0.308 0.393 0.244 0.393 0.247 0.393 0.245 0.251 0.108 0.250 0.109 0.252 0.
1W 0.336 0.185 0.465 0.337 0.397 0.251 0.397 0.255 0.397 0.253 0.338 0.192 0.340 0.193 0.336 0.
2W 0.371 0.219 0.468 0.319 0.404 0.261 0.405 0.266 0.404 0.263 0.374 0.221 0.378 0.232 0.375 0.
1M 0.412 0.272 0.590 0.558 0.418 0.281 0.420 0.286 0.419 0.284 0.415 0.268 0.412 0.273 0.414 0.
```
```
NASDAQ
```
```
1D 0.288 0.144 0.599 0.608 0.706 0.793 0.685 0.682 0.691 0.724 0.310 0.172 0.296 0.157 0.305 0.
3D 0.320 0.198 0.696 0.779 0.708 0.794 0.686 0.684 0.692 0.724 0.385 0.264 0.359 0.226 0.346 0.
1W 0.421 0.309 0.786 1.020 0.707 0.794 0.685 0.682 0.690 0.723 0.473 0.390 0.438 0.346 0.430 0.
2W 0.498 0.420 0.755 0.965 0.705 0.793 0.679 0.676 0.686 0.718 0.544 0.526 0.538 0.520 0.515 0.
1M 0.567 0.571 0.890 1.331 0.703 0.787 0.675 0.665 0.683 0.711 0.632 0.686 0.640 0.711 0.580 0.
```
```
EUR-USD
```
```
1D 0.118 0.023 0.338 0.151 0.171 0.048 0.173 0.049 0.164 0.043 0.116 0.024 0.114 0.023 0.128 0.
3D 0.125 0.027 0.359 0.174 0.172 0.049 0.174 0.049 0.165 0.044 0.124 0.027 0.123 0.026 0.132 0.
1W 0.131 0.030 0.391 0.206 0.175 0.050 0.176 0.050 0.167 0.045 0.133 0.032 0.132 0.031 0.140 0.
2W 0.136 0.033 0.371 0.192 0.178 0.052 0.180 0.052 0.171 0.046 0.137 0.033 0.136 0.033 0.139 0.
1M 0.148 0.039 0.268 0.113 0.195 0.066 0.197 0.066 0.188 0.058 0.149 0.040 0.147 0.039 0.150 0.
```
```
Gold
```
```
1D 0.244 0.100 0.684 0.595 0.360 0.200 0.350 0.191 0.369 0.207 0.274 0.123 0.278 0.126 0.279 0.
3D 0.263 0.127 0.744 0.713 0.361 0.200 0.347 0.189 0.369 0.207 0.299 0.149 0.302 0.152 0.300 0.
1W 0.297 0.164 0.774 0.778 0.362 0.200 0.349 0.190 0.370 0.207 0.323 0.175 0.326 0.179 0.330 0.
2W 0.315 0.170 0.758 0.735 0.362 0.201 0.349 0.190 0.370 0.208 0.318 0.172 0.321 0.175 0.320 0.
1M 0.310 0.166 0.462 0.341 0.355 0.194 0.344 0.185 0.362 0.200 0.311 0.166 0.311 0.168 0.317 0.
```
```
Table 5: Overall comparison of approaches concerning multiple horizons. The bold number indicates the best performance.
```
```
Figure 2: VaR on NASDAQ dataset.
```
NASDAQ dataset. Figures 2 (a) and (b) display the upper
and lower violation rates of 5% 1-day VaR of 1. 65 ∗σˆt
(μt= 0) in NASDAQ out-of-sample period. X-axis ranges
from 2021-02-05 (x= 0) to 2022-07-08 (x= 350), and
Y-axis denotes the returnεt. We can observe that all three
approaches’ total violation rates (upper+lower) are around
5% which indicates the correctness of forecasting the volatil-
ity. We can also see GARCH(1,1) is a strong baseline which
is consistent with the existing literature (Hansen and Lunde
2005) and this explains why it is widely used in the real-
world financial industry. It is also worth noting that this
experiment favors approaches forecasting largeσˆtsince it

```
leads to the large confidence interval that broadens the vi-
olation boundary and in turn loosens the violation criteria.
Thus, the experiment results shown in Figure 2 should not
be analyzed independently. Instead, it should be accompa-
nied by the experiment results shown in Table 5.
```
## Conclusion

```
In this paper, we explore the equivalence relation between
GARCH models and NN. Leveraging this relation, we pro-
pose a novel GARCH-NN approach for devising NN-based
volatility models. This involves deriving the NN equivalents
of GARCH family models, treating them as fundamental
building blocks, and seamlessly integrating them into an es-
tablished NN framework. This method allows the volatil-
ity stylized facts to be seamlessly infused into NN. We de-
velop the GARCH-LSTM model to exemplify the GARCH-
NN approach. Experiment results validate the GARCH-NN
equivalence relation and show that combining the funda-
mental stochastic (GARCH family models) and NN (LSTM)
models yields improved results compared to employing each
model in isolation. For future work, we plan to extend the
current research in the following directions: (1) We plan
to explore if the GARCH-NN equivalence relation widely
exists in the GARCH family beyond the three representa-
tive GARCH models mentioned in this paper to incorporate
more SFs into the NN framework. (2) We plan to integrate
the GARCH kernel into more NN frameworks and study if
it would lead to better volatility modeling.
```

## Acknowledgements

The research reported in this paper was supported in part
by the Guangdong Provincial Key Laboratory of Inter-
disciplinary Research and Application for Data Science,
BNU-HKBU United International College, project code
2022B1212010006, and in part by Guangdong Higher Edu-
cation Upgrading Plan (2021-2025) of “Rushing to the Top,
Making Up Shortcomings and Strengthening Special Fea-
tures” with UIC research grant R0400001-22, Guangdong
Higher Education Upgrading Plan (UIC-R0400024-21), Re-
search Grants Council HKSAR GRF (No. 16215019). We
appreciate Dr. Zhefang Zhou from BNU-HKBU United
International College for insightful discussions and the
anonymous reviewers for their helpful comments on the
manuscript.

## References

Andersen, T. G.; Bollerslev, T.; Diebold, F. X.; and Labys, P.

2003. Modeling and forecasting realized volatility.Econo-
metrica, 71(2): 579–625.

Baillie, R.; Bollerslev, T.; and Mikkelsen, H. O. 1996. Frac-
tionally integrated generalized autoregressive conditional
heteroskedasticity.Journal of Econometrics, 74(1): 3–30.

Bollerslev, T. 1986. Generalized autoregressive conditional
heteroskedasticity. Journal of Econometrics, 31(3): 307–
327.

Bollerslev, T. 1987. A Conditionally Heteroskedastic Time
Series Model for Speculative Prices and Rates of Return.
The Review of Economics and Statistics, 69(3): 542–47.

Bucci, A. 2020. Realized volatility forecasting with neural
networks. Journal of Financial Econometrics, 18(3): 502–
531.

Bucci, A.; et al. 2017. Forecasting realized volatility: a re-
view.Journal of Advanced Studies in Finance (JASF), 8(16):
94–138.

Chen, S.-A.; Li, C.-L.; Yoder, N.; Arik, S. O.; and Pfister,
T. 2023. Tsmixer: An all-mlp architecture for time series
forecasting.arXiv preprint arXiv:2303.06053.

Davidson, J. 2004. Moment and Memory Properties of
Linear Conditional Heteroscedasticity Models, and a New
Model. Journal of Business & Economic Statistics, 22(1):
16–29.

Elsayed, S.; Thyssens, D.; Rashed, A.; Jomaa, H. S.; and
Schmidt-Thieme, L. 2021. Do We Really Need Deep Learn-
ing Models for Time Series Forecasting? arXiv:2101.02118.

Galdi, F. C.; and Pereira, L. M. 2007. Value at Risk (VaR)
Using Volatility Forecasting Models: EWMA, GARCH and
Stochastic Volatility. Brazilian Business Review, 4(1): 74–
94.

Ge, W.; Lalbakhsh, P.; Isai, L.; Lenskiy, A.; and Suominen,
H. 2022. Neural Network–Based Financial Volatility Fore-
casting: A Systematic Review.ACM Comput. Surv., 55.

Giles, C. L.; Lawrence, S.; and Tsoi, A. C. 2001. Noisy
Time Series Prediction using Recurrent Neural Networks
and Grammatical Inference.Mach. Learn.

```
Glosten, L. R.; Jagannathan, R.; and Runkle, D. E. 1993. On
the Relation between the Expected Value and the Volatility
of the Nominal Excess Return on Stocks. Journal of Fi-
nance, 48(5): 1779–1801.
Hansen, P. R.; and Lunde, A. 2005. A forecast comparison of
volatility models: does anything beat a GARCH(1,1)?Jour-
nal of applied econometrics (Chichester, England), 20(7):
873–889.
Hu, Y.; Ni, J.; and Wen, L. 2020. A hybrid deep learning ap-
proach by integrating LSTM-ANN networks with GARCH
model for copper price volatility prediction.Physica A: Sta-
tistical Mechanics and its Applications, 557: 124907.
Kakade, K.; Jain, I.; and Mishra, A. 2022. Value-at-Risk
forecasting: A hybrid ensemble learning GARCH-LSTM
based approach.Resources Policy, 78: 102903.
Khan, S.; Hasanabadi, H. S.; and Mayorga, R. 2017. De-
termining the relationship between speculative activity and
crude oil price volatility, using artificial neural networks. In
2017 International Conference on Information and Commu-
nication Technologies (ICICT), 138–144.
KIlIc ̧, R. 2011. Long memory and nonlinearity in con-
ditional variances: A smooth transition FIGARCH model.
Journal of Empirical Finance, 18(2): 368–378.
Kim, H.; and Won, C. 2018. Forecasting the volatility of
stock price index: A hybrid model integrating LSTM with
multiple GARCH-type models.Expert Systems with Appli-
cations, 103: 25–37.
Kitaev, N.; Kaiser, Ł.; and Levskaya, A. 2020. Reformer:
The efficient transformer.arXiv preprint arXiv:2001.04451.
Kristjanpoller, W.; Fadic, A.; and Minutolo, M. C. 2014.
Volatility Forecast Using Hybrid Neural Network Models.
Expert Syst. Appl., 41(5): 2437–2442.
Kristjanpoller, W.; and Minutolo, M. C. 2016. Forecast-
ing volatility of oil price using an artificial neural network-
GARCH model.Expert Systems with Applications, 65: 233–
241.
Lai, G.; Chang, W.-C.; Yang, Y.; and Liu, H. 2018. Modeling
long-and short-term temporal patterns with deep neural net-
works. InThe 41st international ACM SIGIR conference on
research & development in information retrieval, 95–104.
Li, M.; Li, W.; and Li, G. 2013. On Mixture Memory Garch
Models.Journal of Time Series Analysis, 34.
Lin, H.; and Sun, Q. 2021. Financial Volatility Forecasting:
A Sparse Multi-Head Attention Neural Network. Informa-
tion, 12(10).
Liu, W. K.; and So, M. K. P. 2020. A GARCH Model with
Artificial Neural Networks.Information, 11(10).
Makridakis, S.; Spiliotis, E.; and Assimakopoulos, V. 2018.
Statistical and Machine Learning forecasting methods: Con-
cerns and ways forward.PLOS ONE, 13(3): 1–26.
Masset, P. 2011. Volatility Stylized Facts.SSRN Electronic
Journal.
Oreshkin, B. N.; Carpov, D.; Chapados, N.; and Ben-
gio, Y. 2019. N-BEATS: Neural basis expansion analy-
sis for interpretable time series forecasting.arXiv preprint
arXiv:1905.10437.
```

Pyo, S.; and Lee, J. 2018. Exploiting the low-risk anomaly
using machine learning to enhance the Black–Litterman
framework: Evidence from South Korea. Pacific-Basin Fi-
nance Journal, 51: 1–12.

Rahimikia, E.; and Poon, S.-H. 2020. Machine learning for
realised volatility forecasting.Available at SSRN, 3707796.

Ramos-P ́erez, E.; Alonso-Gonz ́alez, P. J.; and Nu ́nez- ̃
Velazquez, J. J. 2021. Multi-transformer: A new neural ́
network-based architecture for forecasting S&P volatility.
Mathematics, 9(15): 1794.

Salinas, D.; Flunkert, V.; Gasthaus, J.; and Januschowski, T.

2020. DeepAR: Probabilistic forecasting with autoregres-
sive recurrent networks.International Journal of Forecast-
ing, 36(3): 1181–1191.

Sen, R.; Yu, H.-F.; and Dhillon, I. S. 2019. Think glob-
ally, act locally: A deep neural network approach to high-
dimensional time series forecasting.Advances in neural in-
formation processing systems, 32.

Sheppard, K.; Khrapov, S.; Lipt ́ak, G.; mikedeltalima;
Capellini, R.; alejandro cermeno; Hugle; esvhd; bot, S.;
Fortin, A.; JPN; Judell, M.; Li, W.; Adams, A.; jbrock-
mendel; Rabba, M.; Rose, M. E.; Tretyak, N.; Rochette, T.;
Leo, U.; RENE-CORAIL, X.; Du, X.; and C ̧ elik, B. 2022.
bashtage/arch: Release 5.3.1.

Tjoa, E.; and Guan, C. 2021. A Survey on Explainable Arti-
ficial Intelligence (XAI): Toward Medical XAI.IEEE Trans-
actions on Neural Networks and Learning Systems, 32.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones,
L.; Gomez, A. N.; Kaiser, L. u.; and Polosukhin, I. 2017. At-
tention is All you Need. InAdvances in Neural Information
Processing Systems, volume 30.

Wu, H.; Xu, J.; Wang, J.; and Long, M. 2021. Autoformer:
Decomposition transformers with auto-correlation for long-
term series forecasting. Advances in Neural Information
Processing Systems, 34: 22419–22430.

Zeng, A.; Chen, M.; Zhang, L.; and Xu, Q. 2022.
Are Transformers Effective for Time Series Forecasting?
arXiv:2205.13504.

Zheng, J.; Xia, A.; Shao, L.; Wan, T.; and Qin, Z. 2019.
Stock Volatility Prediction Based on Self-attention Net-
works with Social Information. 2019 IEEE Conference
on Computational Intelligence for Financial Engineering &
Economics (CIFEr), 1–7.

Zhou, H.; Zhang, S.; Peng, J.; Zhang, S.; Li, J.; Xiong, H.;
and Zhang, W. 2021. Informer: Beyond Efficient Trans-
former for Long Sequence Time-Series Forecasting.ArXiv,
abs/2012.07436.


