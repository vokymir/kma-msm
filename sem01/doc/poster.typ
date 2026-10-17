// ------------------------------ SETTINGS ------------------------------------

#let logo-path   = "graphics/fav_logo/FAV_cmyk.pdf" // use none to hide it
#let logo-height = 5.5cm 

#let poster-title  = "Vícerozměrné náhodné veličiny"
#let poster-author = "Jakub Vokoun"
#let poster-affil  = "Fakulta aplikovaných věd Západočeské univerzity"
#let poster-date = "17.10.2026"

// A2 = 14pt, A1 = 20pt, A0 = 28pt
#let paper     = "a2"
#let base-size = 14pt

#let accent = rgb(201,151,0)        
#let ink    = rgb("#111111")        

// Fonts (first one found is used). 
// Math always uses New Computer Modern (LaTeX look).
#let sans       = ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")
#let body-font  = sans  // for a LaTeX-like serif body: "New Computer Modern"
#let head-font  = sans

// Spacing
#let col-gutter = 1.6cm // gap between the two columns
#let title-size = 3.4em

// --------------------------- TEMPLATE (don't touch) -------------------------
// * should be in another file, but I am too lazy for it

#set page(paper: paper, margin: (x: 1.8cm, top: 1.5cm, bottom: 1.5cm))
#set text(font: body-font, size: base-size, fill: ink, lang: "cs")
// Want English figure captions ("Figure 1:")? Uncomment next line and use lang: "en" above.
// #set figure(supplement: [Figure])
#set par(justify: true, leading: 0.62em, spacing: 0.95em)

// Math: LaTeX-like font, slightly bigger display equations
#show math.equation: set text(font: "New Computer Modern Math")
#show math.equation.where(block: true): set text(size: 1.1em)
#set math.equation(numbering: "(1)")
#show math.equation.where(block: true): set block(above: 1.1em, below: 1.1em)

// Section bar:  = Title
#show heading.where(level: 1): it => block(
  width: 100%,
  fill: accent,
  inset: (x: 0.6em, y: 0.6em),
  above: 1.9em,
  below: 1.1em,
  align(center, text(font: head-font, fill: white, weight: "bold", size: 1.3em, it.body)),
)
// Sub-heading:  == Title
#show heading.where(level: 2): it => block(
  above: 1.3em,
  below: 0.7em,
  text(font: head-font, weight: "bold", fill: accent.darken(25%), size: 1.1em, it.body),
)

// Figures: bold small captions
#set figure(gap: 0.8em)
#show figure: set block(above: 1.2em, below: 1.2em)
#show figure.caption: set text(font: head-font, size: 0.8em, weight: "bold")

// Fake small-caps (works with any font): first letter of each word big, the rest smaller
#let fake-smallcaps(s, ratio: 0.78) = s.split(" ").map(w => {
  if w == "" { return "" }
  let first = w.clusters().first()
  text(upper(first)) + text(size: ratio * 1em, upper(w.slice(first.len())))
}).join(" ")

// Header: title / author / affiliation on the left, logo on the right, rule underneath
#let poster-header() = {
  set par(justify: false)
  grid(
    columns: (1fr, auto),
    column-gutter: 2em,
    align: (left + horizon, right + horizon),
    {
      text(font: head-font, size: title-size, weight: "bold", fake-smallcaps(poster-title))
      v(0.5em)
      text(size: 1.7em)[#poster-author#super[1]]
      linebreak()
      text(size: 1.15em)[#super[1]#poster-affil]
      linebreak()
      v(0.9em)
      text()[#poster-date]
    },
    if logo-path != none { image(logo-path, height: logo-height) },
  )
  line(length: 100%, stroke: 0.16em + ink)
  v(0.4em)
}

// Grey box for a figure you haven't made yet – delete when you have real images
#let placeholder(h: 6cm, label: "obrázek") = rect(
  width: 100%, height: h, fill: luma(238), stroke: luma(190),
  align(center + horizon, text(fill: luma(120), label)),
)


// -------------------------------- START HERE ---------------------------------

#poster-header()

#columns(2, gutter: col-gutter)[

  = Úvod #footnote([V této práci byla použita umělá inteligence jako nástroj pro
  konzultaci a také aby vygenerovala `python` funkce pro vizualizaci zpracovaných dat.], numbering: "*")

  Uvažujme náhodné vícerozměrné rozdělení dané sdruženou funkcí hustoty

  $
    f(x,y) = c dot x dot y^2
  $ <eq:fx>

  na dvourozměrné oblasti 

  $
    Omega = {(x,y) in R^2 | 0 < x < 2, x+2 < y < 4-x} .
  $ <eq:omega>

  Pro toto rozdělení analyticky odvodíme teoretické charakteristiky,
  vygenerujeme příslušný náhodný výběr (v jazyku `Python` s pomocí knihoven
  `numpy`, `sympy`, `scipy` a `matplotlib`), spočteme jeho charakteristiky a
  porovnáme je.

  = Neznámá konstanta $c$

  @eq:fx může být hustotou pravděpodobnosti pouze tehdy,
  pokud její integrál přes danou oblast bude roven jedné, tedy
  $
    integral_0^1 integral_0^(x+2) c dot x dot y^2 dif y dif x +
    integral_1^2 integral_0^(4-x) c dot x dot y^2 dif y dif x
    = 1.
  $

  Po vyřešení dostáváme hodnotu konstanty $c=6/65$.


  #grid(
    columns: (1.15fr, 1fr), column-gutter: 1em, row-gutter: 1.4em,
    figure(image("graphics/graphs/02_density_3d.svg"), caption: [$f(x, y)$ ve 3D]),
    figure(image("graphics/graphs/14_sample_intervals.svg"), 
    caption: [Náhodně vygenerované body]),
  )

  = Marginální rozdělení

  == Funkce hustoty

  Získáme integrací sdružené funkce hustoty přes okraj její oblasti.

  $
    f_X (x) = cases(
      0 <= x <= 1 #h(1cm) & integral_0^(x+2) f(x,y) dif y = 2/65 x (x+2)^3,
      1 < x <= 2 & integral_0^(4-x) f(x,y) dif y = -2/65 x (x-4)^3,
      "jinak" & 0
    )
  $

  $
    f_Y (y) = cases(
      0 <= y <= 2 #h(1cm) & integral_0^(2) f(x,y) dif x = 12/65 y^2,
      2 < y <= 3 & integral_(y-2)^(4-y) f(x,y) dif x = 12/65 (3-y) y^2,
      "jinak" & 0
    )
  $


// F_Y(y) = Piecewise((0, y <= 0), (4*y**3/65, y <= 2), (-3*y**4/65 + 12*y**3/65 - 16/65, y <= 3), (1, True))

  == Distribuční funkce

  Získáme integrací odpovídajících marginálních funkcí hustoty.

  $
    F_X (x) = integral_0^x f_X (x) dif x =  cases(
      x <= 0 #h(1cm) & 0,
      x <= 1 & 1/325 x^2 (2 x^3 + 15 x^2 + 40 x + 40),
      // x <= 2 & -2/325 x^5 + 6/65 x^4 - 32/65 x^3 + 64/65 x ^2 - 7/25,
      x <= 2 & 1/325 (-2 x^5 + 30 x^4 - 160 x^3 + 320 x ^2 - 91),
      x > 2 & 1,
    )
  $

  $
    F_Y (y) = integral_0^y f_Y (y) dif y =  cases(
      y <= 0 #h(1cm) & 0,
      y <= 2 & 4/65 y^3,
      // y <= 3 & -3/65 y^4 + 12/65 y^3 - 16/65,
      y <= 3 & 1/65 (-3 y^4 + 12 y^3 - 16),
      y > 3 & 1,
    )
  $

  = Teoretické charakteristiky

  $
    E[X] &= integral.double_Omega x f(x,y) dif x dif y &&= 1208/975 #h(1.5cm) &&(~1.24) \
    E[Y] &= integral.double_Omega y f(x,y) dif y dif x &&= 633/325 &&(~1.95) \
  $
  // $
  //   "Var"(X) &= E[X^2] - (E X)^2 &&= 172886/950625 &&(~0.18) \
  //   "Var"(Y) &= E[Y^2] - (E Y)^2 &&= 31561/105625 &&(~0.30) \
  //   "Cov"(X, Y) &= E(X Y) - E(X) E(Y) &&= -38966/739375 &&(~-0.05) \
  // $

  $
    Sigma = mat(
      "var"(X), "cov"(X,Y);
      "cov"(X,Y), "var"(Y);
    )
    =
    mat(
      172886/950625, -38966/739375;
      -38966/739375, 31561/105625
    )
  $

  = Generování dat

  Generování náhodného výběru se řídilo zamítací metodou. Bylo vygenerováno na
  1000 bodů s fixním seedem generátoru náhodných čísel pro jednoduchou
  replikaci.
  
  #grid(
    columns: (1fr, 1fr), column-gutter: 1em, row-gutter: 1.4em,
    figure(image("graphics/graphs/15_hist_ecdf_x.svg"),
      caption: [Marginální hustota a distribuce $X$]),
    figure(image("graphics/graphs/16_hist_ecdf_y.svg"), 
      caption: [Marginální hustota a distribuce $Y$]),
  )

  Už pro náhodný výběr velikosti $10^3$ lze vidět, že se hustota i distribuce
  velmi blíží analyticky získaným hodnotám.

  = Empirické charakteristiky

  Odhady z výběru a jejich 95% intervalové odhady

  $
    hat(x) &= 1.222 plus.minus 0.026 \
    hat(y) &= 1.982 plus.minus 0.034 \
    "var"(x) &= 0.171 plus.minus 0.015 \
    "var"(y) &= 0.300 plus.minus 0.026 \
    "cov"(X,Y) &= -0.047.
  $

  = Přesnost odhadu v závislosti na počtu dat

  Testováním na náhodných výběrech o velikostech 50, 100, 250, 500, 1000, 2500 a
  5000 ukazujeme závislost přesnosti na velikosti, konkrétně se odhady s větší
  velikostí zpřesňují.

  #grid(
    columns: (1fr, 1fr), column-gutter: 1em, row-gutter: 1.4em,
    figure(image("graphics/graphs/17_size_estimates_all.svg"),
      caption: [Závislost odhadů charakteristik rozdělení (normalizované podle
      analyticky získaných hodnot) na velikosti náhodného výběru.]),
    figure(image("graphics/graphs/18_size_widths_all.svg"), 
      caption: [Závislost šířek intervalových odhadů na velikosti náhodného
      výběru.]),
  )

]
