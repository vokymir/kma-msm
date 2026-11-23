// ------------------------------ SETTINGS ------------------------------------

#let logo-path   = "graphics/fav_logo/FAV_cmyk.pdf" // use none to hide it
#let logo-height = 4.2cm

#let poster-title  = "Testy o poloze: okoun, cejn a plotice"
#let poster-author = "Jakub Vokoun"
#let poster-affil  = "Fakulta aplikovaných věd Západočeské univerzity"
#let poster-date   = "23.11.2026"

// A2 = 14pt, A1 = 20pt, A0 = 28pt
#let paper     = "a2"
#let base-size = 13pt

#let accent = rgb(201,151,0)
#let ink    = rgb("#111111")

// Fonts (first one found is used).
#let sans       = ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")
#let body-font  = sans
#let head-font  = sans

// Spacing
#let col-gutter = 1.6cm
#let title-size = 2.6em

// Cesta k obrázkům z programu (main.py je ukládá do vystupy/)
#let img-dir = "vystupy/"


// --------------------------- TEMPLATE (don't touch) -------------------------

#set page(paper: paper, margin: (x: 1.8cm, top: 1.5cm, bottom: 1.5cm))
#set text(font: body-font, size: base-size, fill: ink, lang: "cs")
#set par(justify: true, leading: 0.62em, spacing: 0.95em)

#show math.equation: set text(font: "New Computer Modern Math")
#show math.equation.where(block: true): set text(size: 1.1em)
#set math.equation(numbering: "(1)")
#show math.equation.where(block: true): set block(above: 1.1em, below: 1.1em)

#show heading.where(level: 1): it => block(
  width: 100%,
  fill: accent,
  inset: (x: 0.6em, y: 0.45em),
  above: 1.2em,
  below: 0.8em,
  align(center, text(font: head-font, fill: white, weight: "bold", size: 1.3em, it.body)),
)
#show heading.where(level: 2): it => block(
  above: 0.9em,
  below: 0.5em,
  text(font: head-font, weight: "bold", fill: accent.darken(25%), size: 1.1em, it.body),
)

#set figure(gap: 0.8em)
#show figure: set block(above: 0.8em, below: 0.8em)
#show figure.caption: set text(font: head-font, size: 0.8em, weight: "bold")

#let fake-smallcaps(s, ratio: 0.78) = s.split(" ").map(w => {
  if w == "" { return "" }
  let first = w.clusters().first()
  text(upper(first)) + text(size: ratio * 1em, upper(w.slice(first.len())))
}).join(" ")

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

// Tabulka s jemnějším vzhledem
#let nice-table(..args) = {
  set text(size: 0.85em)
  table(
    stroke: (x, y) => if y == 0 { (bottom: 0.1em + ink) } else { (bottom: 0.03em + luma(190)) },
    inset: (x: 0.6em, y: 0.35em),
    align: (left, center, center, center, center),
    ..args,
  )
}

// -------------------------------- START HERE ---------------------------------

#poster-header()

#columns(2, gutter: col-gutter)[

  = Úvod

  Porovnáváme *polohu* dvou spojitých znaků – *hmotnosti* (`Weight`, g) a
  *výšky těla* (`Height`, cm) – u tří druhů ryb z trhu: *okoun* (`Perch`),
  *cejn* (`Bream`) a *plotice* (`Roach`). Provádíme jedno-, dvou-,
  vícevýběrové a vícerozměrné testy polohy, vždy parametricky i
  neparametricky (kromě vícerozměrné části).

  - *Data:* Fish Market (veřejný CSV, 159 ryb, 7 druhů). Použity tři druhy;
    po vyřazení jedné plotice s nulovou hmotností (podezřelé měření) zbývá
    $n = 110$: okoun 56, cejn 35, plotice 19.
  - *Technologie:* `Python` 3 (`pandas`, `numpy`, `scipy`, `statsmodels`,
    `matplotlib`, `seaborn`).
  - *Použití AI:* konzultace postupu a programování (načítání dat, pomocné
    funkce, grafy); volbu metod, kontrolu a interpretaci provedl autor.

  = Použité metody

  #nice-table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    [*Úroveň*], [*Parametricky*], [*Neparametricky*],
    [1 výběr], [t-test], [znaménkový test],
    [2 výběry], [Welchův t-test], [Mann–Whitney U],
    [$k$ výběrů], [ANOVA + Tukey HSD], [Kruskal–Wallis + Dunn],
    [vícerozměrně], [MANOVA, Hotelling $T^2$], [—],
    [předpoklady], [Shapiro–Wilk (rezidua), Levene, Box M], [—],
  )

  *Korekce:* Holm, zvlášť pro každou proměnnou a typ testu (Tukey má vlastní
  korekci), $alpha = 0.05$.

  *Jednovýběrové:* $H_0: mu_"druh" = mu_0$ vs. $H_1: mu_"druh" != mu_0$,
  kde $mu_0$ je průměr (medián) všech tří druhů. Skupina je součástí reference,
  proto jsou p-hodnoty *demonstrační*, ne nezávislé potvrzení hypotézy.

  *Vícerozměrně:* MANOVA (Pillaiova stopa) testuje shodu vektorů středních
  hodnot, pro dvojice druhů Hotellingův test.

  *Omezení metod:* Tukeyho test a (sdružený) Hotellingův $T^2$ předpokládají
  shodu rozptylů, resp. kovariančních matic, což Levene i Box M zamítají.
  Berte je proto jako orientační a více důvěřujte Welchovu testu, Dunnovu testu
  a Pillaiově stopě.

  = Výsledky

  #figure(
    image(img-dir + "01_rozdeleni.png"),
    caption: [Hmotnost a výška podle druhu. Přerušovaně celkový průměr,
    tečkovaně celkový medián.],
  )

  #nice-table(
    columns: (1.3fr, 0.6fr, 1.2fr, 1.2fr),
    align: (left, center, center, center),
    [*Druh*], [*n*], [*průměr / medián [g]*], [*průměr / medián [cm]*],
    [Okoun], [56], [382.2 / 207.5], [7.86 / 6.92],
    [Cejn], [35], [617.8 / 610.0], [15.18 / 14.95],
    [Plotice], [19], [160.1 / 150.0], [6.71 / 6.55],
    [Celkem (reference)], [110], [418.8 / 300.0], [9.99 / 9.18],
  )

  #v(0.8em)

  #nice-table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    [*Druh*], [*Hmotnost, t*], [*Hmotnost, znam.*], [*Výška, t*], [*Výška, znam.*],
    [Okoun], [$0.434$], [$0.076$], [$8.86 times 10^(-7)$], [$0.022$],
    [Cejn], [$5.25 times 10^(-6)$], [$1.10 times 10^(-7)$], [$1.30 times 10^(-16)$], [$1.75 times 10^(-10)$],
    [Plotice], [$2.21 times 10^(-10)$], [$1.53 times 10^(-4)$], [$3.70 times 10^(-9)$], [$1.53 times 10^(-4)$],
  )

  #text(size: 0.75em)[_Jednovýběrové testy proti celku, Holm-korigované p-hodnoty (t = t-test, znam. = znaménkový test)._]

  - *Jednovýběrové testy:* cejn má významně vyšší hmotnost i výšku než celek,
    plotice významně nižší obojí. Okoun má významně *nižší* výšku (průměr
    7.86 vs. 9.99 cm), ale jeho hmotnost se od celku neliší ($p = 0.434$,
    znaménkový test $p = 0.076$). T-test i znaménkový test dávají ve všech
    šesti případech stejný závěr.

  #figure(
    image(img-dir + "03_parova_porovnani.png", width: 100%),
    caption: [Holm-korigované p-hodnoty Welchova testu pro všechny dvojice.],
  )


  #figure(
    image(img-dir + "02_vicerozmerny.png", width: 70%),
    caption: [Hmotnost vs. výška, 95% elipsy rozptylu dat jednotlivých druhů.
  Elipsy ukazují odlišnou kovarianční strukturu druhů; Boxův M test:
  $p = 8.29 times 10^(-12)$.
  ],
  )


- *Post-hoc a párové testy:*
  - *Cejn* se od okouna i plotice liší v obou proměnných ve všech čtyřech testech (Welch, Tukey, Mann–Whitney, Dunn).
  - *Okoun × plotice* je nestabilní (tabulka níže). V hmotnosti ji nepotvrdil jen Mann–Whitney ($p = 0.062$), ve výšce ji potvrdil jen Welch ($p = 0.020$).
  - Možné vysvětlení u hmotnosti: okounovo rozdělení je pravostranně zešikmené, velké ryby zvedají průměr, ale ne pořadí hodnot, které Mann–Whitney porovnává.

#nice-table(
  columns: (1.2fr, 1fr, 1fr, 1fr, 1fr),
  [*Okoun – Plotice*], [*Welch*], [*Tukey*], [*Mann–Whitney*], [*Dunn*],
  [Hmotnost], [$7.10 times 10^(-5)$], [$0.009$], [$0.062$], [$0.019$],
  [Výška], [$0.020$], [$0.170$], [$0.321$], [$0.462$],
)

#text(size: 0.75em)[_Holm-korigované p-hodnoty (Tukey: vlastní korekce)._]

#figure(
  image(img-dir + "06_okoun_plotice.png", width: 60%),
  caption: [Okoun a plotice: rozdělení hmotnosti a výšky. Přerušovaně
  průměr, tečkovaně medián.],
)

#v(0.4em)

- *MANOVA (Pillai):* všechny dvojice druhů se při společném hodnocení hmotnosti a výšky významně liší.

- *Předpoklady:* Shapiro–Wilkův test reziduí i Leveneův test ukazují na porušení
  normality i shody rozptylů. Welchův test řeší jen nestejné rozptyly, normalitu
  stále předpokládá. U okouna a cejna ($n = 56, 35$) ji zmírňuje větší počet
  pozorování, u plotic ($n = 19$) méně. Parametrické výsledky proto doplňujeme
  neparametrickými (Mann–Whitney, Kruskal–Wallis, Dunn) jako kontrolou.

#nice-table(
  columns: (1.1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center, center, center, center),
  [*Proměnná*], [*ANOVA $F$*], [*ANOVA $p$*], [*$eta^2$*], [*K–W $H$*], [*K–W $p$*], [*$epsilon^2$*],
  [Hmotnost],
  [$17.704$], [$2.28 times 10^(-7)$], [$0.249$], [$30.38$], [$2.53 times 10^(-7)$], [$0.279$],
  [Výška],
  [$121.598$], [$2.83 times 10^(-28)$], [$0.694$], [$68.81$], [$1.14 times 10^(-15)$], [$0.631$],
)

#text(size: 0.75em)[_$eta^2$ z ANOVY, $epsilon^2 = H slash (n - 1)$ z Kruskalova–Wallisova testu, $n = 110$._]

#v(0.4em)

== Proč více metod

- *Welch místo klasického t-testu:* rozptyly druhů se liší, a Welchův test
  shodu rozptylů nevyžaduje.
- *Neparametrické testy jako kontrola:* Welch normalitu stále předpokládá. 
  Hmotnost okouna je zešikmená a pořadové testy jsou méně citlivé na pár velkých ryb.
- *Holmova korekce:* každou proměnnou porovnáváme ve třech dvojicích,
  korekce brání tomu, aby se nahromadily falešně významné výsledky.
- *Vícerozměrný pohled:* hmotnost a výška spolu silně souvisejí (protáhlé
  elipsy na obrázku 3), proto je hodnotí MANOVA a Hotelling společně.
- *p-hodnota vs. velikost efektu:* p-hodnota říká, zda rozdíl existuje,
  $eta^2$ a $epsilon^2$ říkají, jak je velký. Druh vysvětluje většinu
  variability výšky, ale jen menší část variability hmotnosti.

= Závěr

- Druhy ryb se statisticky významně liší v hmotnosti i výšce těla (Kruskal–Wallis, MANOVA).
  Nejvýraznější rozdíly jsou mezi cejnem a ostatními druhy, kde se shodují
  všechny čtyři párové testy. Efekt druhu je větší u výšky těla
  ($epsilon^2 = 0.631$) než u hmotnosti ($epsilon^2 = 0.279$).

- Parametrické a neparametrické metody se shodují u všech dvojic s cejnem.
  Nestabilní je dvojice okoun–plotice: v hmotnosti ji nepotvrdil jen
  Mann–Whitney, ve výšce ji potvrdil jen Welch. Rozdíl hmotnosti proto
  považujeme za spíše pravděpodobný (3 ze 4 testů; Mann–Whitney je těsně nad
  hranicí a rozdíl mediánů je jen 57.5 g), rozdíl výšky za neprokázaný. Okoun má navíc významně nižší výšku
  než průměr všech ryb, kdežto jeho hmotnost se od průměru neliší.

- Výsledky je nutné interpretovat s ohledem na omezení analýzy: referenční
  hodnoty jednovýběrových testů jsou odvozeny ze stejných dat, normalita
  reziduí i shoda rozptylů a kovariančních matic jsou porušeny (Tukey a
  sdružený Hotelling $T^2$ jsou proto jen orientační) a skupina plotic je
  malá ($n = 19$). Jde navíc o observační data, která neumožňují vyvozovat
  příčinné vztahy.

]
