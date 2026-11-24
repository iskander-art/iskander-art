# The Nour Gallery: mihrab reel

An organic Instagram reel (1080×1920, 30 fps, ~29 s) for @thenourgallery. It has:
- an Algerian Darija voiceover by a female voice (ElevenLabs);
- still images animated with code;
- phrase-by-phrase captions with gold highlights;
- synthesized sound effects;
- a branded end card.

## Script (voice: "Nour Algerian Darija (design 2)", model eleven_v4)

| Time | Line | Shot |
|------|------|------|
| 0.0 | عندك حيط فارغ فالدار؟ | first panel going up on the bare wall |
| 1.9 | خليه يولّي ركن الصلاة تاعك! | second panel |
| 4.3 | محراب ذهبي… | finished mihrab, medallion going up |
| 5.7 | مقصوص بدقة، وتفاصيلو تلمع مع الضو. | close-up of the real lattice with a moving glint |
| 8.8 | فوقو اسم الله… | الله medallion |
| 11.0 | واسم سيدنا محمد صلى الله عليه وسلم. | محمد medallion |
| 14.4 | ركن صغير… يجيب الراحة والسكينة لدارك كاملة. | mother reading Quran, child running to her |
| 18.3 | بلاصة فين تصلّي العايلة كاملة. | kids praying (customer story, faces covered) |
| 20.7 | حابّة واحد كيما هذا؟ | relaxing with tea by the mihrab (customer story) |
| 22.3 | راسلينا فالخاص! | real corner photo |
| 24.3 | ذا نور غاليري… نوّري دارك. | fairy-light mihrab, white flash on the cut |
| 27.0 | end card | The Nour Gallery · نوّري دارك · راسلينا فالخاص · @thenourgallery |

## Pipeline

```bash
python reel.py --img IMAGES/ --out picture.mp4         # animated picture track (parallel)
node captions.js captions.mov 28.8                      # captions + end card with alpha
python finish.py --voice voice.mp3 --picture picture.mp4 --captions captions.mov --out reel.mp4
```

- **`reel.py`** builds a simple depth map per image (wall, floor, door, and the person in the abaya cut out by darkness) and moves a virtual camera with parallax. It adds handheld micro-motion, a gold glint, twinkling fairy lights and lantern flicker. The three real photos reuse the depth maps from `../mihrab-footage/footage.py`.
- **`captions.html` / `captions.js`** render one phrase at a time in Cairo Black, 84 px, at y = 1250 (clear of Instagram's buttons), with a pop-in and one gold highlight per phrase. They also render the end card. Frames are transparent PNGs rendered by headless Chromium.
- **`finish.py`** synthesizes the sound effects with numpy: a boom at the start, whooshes on the cuts, pops on the highlights, an impact on the brand flash and a shimmer on the end card. It cleans up the voice (high-pass, presence boost, compressor), mixes to −14 LUFS and overlays the captions.

`IMAGES/` must contain `install1.jpg install2.jpg install3.png family.jpg kids_story.jpg tea_story.jpg lights.jpg side.jpg front.png corner.png`. The images, voice and rendered video are not in this repository.

Fonts in `fonts/`: Cairo and Amiri, both under the SIL Open Font License 1.1.
