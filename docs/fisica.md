# Física e processamento de sinais

Este documento descreve o modelo físico e as escolhas de DSP do radar passivo,
e deixa explícito o que é **medida** e o que é **estimativa**.

## 1. Geometria bistática e efeito Doppler

O transmissor (VOR) ilumina a aeronave; o receptor capta a reflexão. Com
$R_T$ a distância VOR→aeronave e $R_R$ a distância aeronave→receptor, o
trajeto bistático é $R_b = R_T + R_R$ e o desvio Doppler do eco vale

$$
\Delta f = -\frac{1}{\lambda}\,\frac{dR_b}{dt}
\qquad\Longrightarrow\qquad
v_b \equiv \frac{dR_b}{dt} = -\lambda\,\Delta f .
$$

* $\Delta f > 0$ ⇒ $v_b < 0$: o trajeto está **encurtando** ("aproximação").
* $v_b$ **não** é a velocidade da aeronave. Como cada perna varia no máximo
  com a velocidade $v$, vale $|v_b| \le 2v$.

### Ordens de grandeza (VOR em 115,9 MHz)

| Grandeza | Valor |
|---|---|
| Comprimento de onda $\lambda = c/f$ | 2,587 m |
| 1 Hz de Doppler equivale a | 2,59 m/s ≈ 5,0 kt de $v_b$ |
| $\lvert\Delta f\rvert$ máximo físico para $v \le 340$ m/s | ≈ 263 Hz |
| Erro do TCXO de 1 ppm | ≈ ±116 Hz |

Os desvios são pequenos (dezenas a centenas de Hz), por isso a análise exige
resolução espectral da ordem de 1 Hz.

## 2. O sinal do VOR e o que o eco carrega

O VOR transmite uma portadora em 108–117,95 MHz modulada em amplitude por:

* tom de **30 Hz** (30 % de profundidade) — no VOR convencional (CVOR) é o
  sinal "variável", cuja fase depende da radial;
* **subportadora de 9960 Hz** (30 %), modulada em frequência por 30 Hz com
  desvio de ±480 Hz — no CVOR é a "referência";
* identificação em Morse com tom de 1020 Hz.

No Doppler-VOR (DVOR), comum no Brasil, os papéis de referência e variável se
invertem, mas a estação é construída para que o receptor de bordo calcule a
radial do mesmo modo.

A aeronave reflete **o sinal que chega até ela**, portanto o eco carrega:

* a portadora deslocada por $\Delta f$ (o traço em "S" no waterfall);
* bandas laterais AM em $\Delta f \pm 30$ Hz, cerca de 16,5 dB abaixo, que
  aparecem como traços fracos paralelos (o rastreador as descarta como
  "fantasmas");
* a fase de 30 Hz correspondente à **radial em que a aeronave está**, que é
  usada pelo módulo experimental de azimute (seção 8).

## 3. Cadeia de DSP

```
RTL-SDR 1,024 MS/s  (LO em f_VOR + 256 kHz, para fugir do spike DC)
  │ mistura por +fs/4  (multiplicação por jⁿ, sem custo)  → VOR em 0 Hz
  │ FIR ↓8   Kaiser, ~70 dB                                → 128 kS/s
  │ FIR ↓4   banda útil ±11 kHz                            →  32 kS/s  (fluxo largo: IQ gravado, azimute)
  │ NCO fino (offset em Hz)
  │ FIR ↓8   banda útil ±1,6 kHz                           →   4 kS/s  (fluxo estreito: Doppler)
  ▼
FFT longa (padrão 4096 pontos, Hann, 75 % de overlap, média de 2 espectros)
```

Todos os FIR são de fase linear e implementados como decimadores polifásicos
com estado entre blocos: só as amostras de saída necessárias são calculadas e
nenhuma amostra é perdida entre blocos.

| FFT @ 4 kS/s | Resolução | Janela | Linhas/s (75 %) |
|---|---|---|---|
| 2048 | 1,95 Hz | 0,51 s | 7,8 |
| **4096** | **0,98 Hz** | **1,02 s** | **3,9** |
| 8192 | 0,49 Hz | 2,05 s | 2,0 |

Janelas longas melhoram a resolução, mas borram o traço quando o Doppler varia
depressa (perto da CPA, $|df/dt|$ chega a vários Hz/s). Com 4096 pontos o
compromisso é bom para a maioria das passagens.

### C/N0 × SNR por bin

A medida física da força do eco é $C/N_0$ (dB-Hz). Numa FFT com janela de
banda equivalente de ruído $B_{eq}$ (Hann, 4096 pontos a 4 kS/s:
$B_{eq} \approx 1{,}47$ Hz), o SNR do pico no bin é

$$
\mathrm{SNR}_{bin} \approx C/N_0 - 10\log_{10} B_{eq} \approx C/N_0 - 1{,}7\ \text{dB}.
$$

A média incoerente de $N$ espectros reduz a flutuação do ruído e melhora a
detectabilidade em cerca de $5\log_{10}N$ dB.

## 4. Referência de 0 Hz Doppler

O $\Delta f$ é medido em relação a uma referência. Sem visada direta, o eixo de
frequência do receptor carrega o erro do oscilador do RTL-SDR (TCXO de 1 ppm ⇒
até ±116 Hz) somado ao erro do transmissor. Em $v_b$ isso pode chegar a
centenas de m/s, então a referência é tratada com cuidado:

* **Referência automática**: um espectro de média longa ($\tau$ = 120 s)
  procura a linha estática dominante em ±300 Hz. Mesmo sem visada, o sinal
  direto costuma chegar por difração ou por clutter de terreno com Doppler
  nulo. Quando a proeminência passa de 5 dB, essa linha vira o 0 Hz.
* **Sem linha estática**: a interface avisa que $\Delta f$ e $v_b$ absolutos
  incluem o erro do LO. Nesse caso calibre o ppm do dongle (por exemplo, numa
  estação LTE) ou use o offset fino.

**Não dependem da referência**: $df/dt$, a forma do "S", o instante $t_0$ do
ajuste e os intervalos de velocidade e distância. Um offset constante é
absorvido pelo parâmetro $c$ do modelo (seção 7).

## 5. Detecção

1. **Remoção de linhas estáticas.** O espectro é dividido por um fundo de
   média exponencial ($\tau$ = 45 s). Isso remove a portadora residual, o
   clutter, as bandas AM de 30 Hz do sinal direto e espúrios fixos, e deixa o
   piso branco. Nos bins onde há alvo, o fundo é atualizado 20× mais devagar,
   para que um alvo lento não seja "absorvido".
2. **CFAR 1D.** O ruído local é estimado em células de referência dos dois
   lados (com células de guarda). O modo OS usa a mediana, corrigida pelo viés
   da distribuição Gama de $N$ espectros médios, e resiste a alvos vizinhos. O
   modo CA usa a média.
3. **Picos.** São selecionados os máximos locais acima do limiar (padrão
   9 dB), fora da zona morta em torno de 0 Hz e das bandas ±30 Hz, com
   interpolação parabólica sub-bin.

## 6. Rastreamento

* Predição linear por regressão nos últimos ~6 s e associação ótima
  (algoritmo húngaro) dentro de uma janela que cresce com $|df/dt|$.
* Confirmação: número mínimo de detecções **e** duração mínima (os quadros com
  overlap são correlacionados, então só contar detecções não basta).
* **Inércia em zonas cegas**: quando a predição cai na zona morta de 0 Hz ou
  das bandas ±30 Hz, a trilha confirmada continua por até 5× a tolerância.
  Isso evita partir a trilha justamente no cruzamento.
* Detecções a ±30 Hz de um pico ≥ 6 dB mais forte são descartadas como
  bandas laterais AM do próprio eco.

## 7. Passagem pelo ponto mais próximo (CPA)

Para voo retilíneo e uniforme, se uma perna domina a variação, ela vale
$R(t) = \sqrt{d^2 + v^2 (t - t_0)^2}$ e a outra contribui com um termo quase
constante. O Doppler fica

$$
\Delta f(t) \approx -A\,\frac{u}{\sqrt{\tau^2 + u^2}} + c,\qquad u = t - t_0,\quad A = \frac{v}{\lambda},\quad \tau = \frac{d}{v}.
$$

O ajuste não linear dos parâmetros $(A,\,t_0,\,\tau,\,c)$ fornece:

| Saída | Significado | Robustez |
|---|---|---|
| $t_0$ | CPA da perna dominante (inflexão do "S") | aproximação |
| $-A/\tau$ | inclinação do Doppler na CPA (Hz/s) | boa |
| $t_{zero}$ | instante em que $\Delta f = 0$, ou seja, mínimo de $R_T+R_R$ | exato se a referência estiver correta |
| $v \in [A\lambda/2,\ A\lambda]$ | velocidade: hipótese quase monostática … perna única | intervalo |
| $d_{CPA} = v\,\tau$ | distância na CPA, nas mesmas hipóteses | intervalo |

O instante de Doppler nulo tem forma fechada: se $|c| < A$, com $q = c/A$,

$$
t_{zero} = t_0 + \tau\,\frac{q}{\sqrt{1-q^2}} .
$$

**Viés conhecido.** Na simulação com o VOR a 25 km e a aeronave passando a
10 km do receptor, as duas pernas variam: $t_0$ ficou ~7 s depois da CPA real
ao receptor, enquanto $t_{zero}$ errou 0,01 s. Sem medir atraso de propagação
(o sinal é CW e não há referência direta), não existe distância absoluta. Por
isso a interface rotula esses valores como estimativas geométricas.

## 8. Azimute experimental (radial pelo eco)

1. O fluxo de 32 kS/s é deslocado pela frequência da trilha (com $df/dt$).
2. A portadora do eco é recuperada com um passa-baixas de 5 Hz e usada como
   referência de fase numa detecção AM síncrona.
3. AM de 30 Hz: lock-in em 30 Hz → fase $\varphi_{AM}$.
4. Subportadora 9960 ± 700 Hz: sinal analítico → frequência instantânea →
   lock-in em 30 Hz → fase $\varphi_{FM}$.
5. Radial = $\varphi_{FM} - \varphi_{AM}$ (+ calibração).

Todos os filtros são de fase zero (filtfilt), porque 1 ms de atraso diferencial
entre os ramos já daria 10,8° de erro. Na simulação, com $C/N_0$ = 50 dB-Hz e
sem sinal direto, o erro ficou abaixo de 0,5°.

Limitações: exige eco forte (a demodulação FM tem limiar), e se o sinal direto
estiver presente, a subportadora dele (±480 Hz de desvio) se sobrepõe à do eco,
que só está deslocada pelo Doppler.

## 9. Simulador

`vorradar/simulator.py` gera IQ amostra a amostra a partir da geometria real
(coordenadas locais com o receptor na origem): $R_T$, $R_R$ e a fase
$-2\pi(R_T+R_R)/\lambda$ de cada aeronave, amplitude pela equação do radar
($\propto 1/(R_T R_R)$), modulação CVOR completa com a radial da aeronave,
resíduo do sinal direto, erro de LO e ruído gaussiano. Os testes em
`tests/test_pipeline.py` comparam a saída do pipeline com essa verdade.
