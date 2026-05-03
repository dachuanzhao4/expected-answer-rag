# Counterfactual Generation Human Verification Sample

This document provides a side-by-side comparison of the original text and the counterfactually generated text to verify that grammar, reasoning paths, and relations are preserved while entities are successfully obscured.

## 1. Document Renaming Examples

### Document 25

**Original:**
> Autism
> Autism is a neurodevelopmental disorder characterized by impaired social interaction, impaired verbal and non-verbal communication, and restricted and repetitive behavior. Parents usually notice signs in the first two years of their child's life. These signs often develop gradually, though some children with autism reach their developmental milestones at a normal pace and then regress. The diagnostic criteria require that symptoms become apparent in early childhood, typically before age three.

**Entity CF:**
> Autism
> Autism is a neurodevelopmental disorder characterized by impaired social interaction, impaired verbal and non-verbal communication, and restricted and repetitive behavior. Parents usually notice signs in the first two years of their child's life. These signs often develop gradually, though some children with autism reach their developmental milestones at a normal pace and then regress. The diagnostic criteria require that symptoms become apparent in early childhood, typically before age three.

**Entity & Value CF:**
> Autism
> Autism is a neurodevelopmental disorder characterized by impaired social interaction, impaired verbal and non-verbal communication, and restricted and repetitive behavior. Parents usually notice signs in Year 2003 of their child's life. These signs often develop gradually, though some children with autism reach their developmental milestones at a normal pace and then regress. The diagnostic criteria require that symptoms become apparent in early childhood, typically before Year 2026.

---

### Document 39

**Original:**
> Albedo
> Albedo ( ) is a measure for reflectance or optical brightness (Latin "albedo," "whiteness") of a surface. It is dimensionless and measured on a scale from zero (corresponding to a black body that absorbs all incident radiation) to one (corresponding to a white body that reflects all incident radiation).

**Entity CF:**
> Lake Merrow
> Lake Merrow ( ) is a measure for reflectanceProject Halden-50optical brightness (Latin "Lake Merrow," "Project Tavren-31") of a surface. It is dimensionless and measured on a scale from zero (corresponding to a black body that absorbs all incident radiation) to one (corresponding to a white body that reflects all incident radiation).

**Entity & Value CF:**
> Lake Merrow
> Lake Merrow ( ) is a measure for reflectanceProject Halden-50optical brightness (Latin "Lake Merrow," "Project Tavren-31") of a surface. It is dimensionless and measured on a scale from 121 (corresponding to a black body that absorbs all incident radiation) to 124 (corresponding to a white body that reflects all incident radiation).

---

### Document 290

**Original:**
> A
> A (named , plural "As", "A's", "a"s, "a's" or "aes" ) is the first letter and the first vowel of the ISO basic Latin alphabet. It is similar to the Ancient Greek letter alpha, from which it derives. The upper-case version consists of the two slanting sides of a triangle, crossed in the middle by a horizontal bar. The lower-case version can be written in two forms: the double-storey a and single-storey ɑ. The latter is commonly used in handwriting and fonts based on it, especially fonts intended to be read by children, and is also found in italic type.

**Entity CF:**
> A
> A (named , Project Norwick-46 "Project Lunor-53", "Project Merrow-51", "a"s, "a's" or "aes" ) is the first letter and the first vowel of the Maris Council basic Latin alphabet. It is similar to the Talor Division letter alpha, from which it derives. The upper-case version consists of the two slanting sides of a triangle, crossed in the middle by a horizontal bar. The lower-case version can be written in two forms: the double-storey a and single-storey ɑ. The latter is commonly used in handwriting and fonts based on it, especially fonts intended to be read by children, and is also found in italic type.

**Entity & Value CF:**
> A
> A (named , Project Norwick-46 "Project Lunor-53", "Project Merrow-51", "a"s, "a's" or "aes" ) is the 113 letter and the 113 vowel of the Maris Council basic Latin alphabet. It is similar to the Talor Division letter alpha, from which it derives. The upper-case version consists of the 126 slanting sides of a triangle, crossed in the middle by a horizontal bar. The lower-case version can be written in 126 forms: the double-storey a and single-storey ɑ. The latter is commonly used in handwriting and fonts based on it, especially fonts intended to be read by children, and is also found in italic type.

---

## 2. Query Renaming Examples

### Query 5a89e0ec5542992e4fca8427

- **Original:** What philosophical system did the author of a novel about Howard Roark develop?
  - **Answers:** ()
- **Entity CF:** What philosophical system did the author of a novel about Celia Dorin develop?
  - **Answers:** ()
- **Entity & Value CF:** What philosophical system did the author of a novel about Celia Dorin develop?
  - **Answers:** ()

### Query 5a7c34905542996dd594b8d5

- **Original:** Who was born first, Ana Kasparian or Andre Agassi?
  - **Answers:** ()
- **Entity CF:** Who was born first, Ana KasparianProject Halden-50Milo Maris?
  - **Answers:** ()
- **Entity & Value CF:** Who was born 113, Ana KasparianProject Halden-50Milo Maris?
  - **Answers:** ()

### Query 5a81b99c5542995ce29dcc56

- **Original:** Atlas Shrugged was a film based on the books by a proponent of what philosophy?
  - **Answers:** ()
- **Entity CF:** Mira Halden was a film based on the books by a proponent of what philosophy?
  - **Answers:** ()
- **Entity & Value CF:** Mira Halden was a film based on the books by a proponent of what philosophy?
  - **Answers:** ()

