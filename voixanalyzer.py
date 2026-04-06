import streamlit as st
import numpy as np
import tempfile, os
from scipy import signal
from scipy.fft import rfft, rfftfreq

EPS = 1e-12

st.set_page_config(
    page_title="VoixAnalyzer — Diagnostic pro",
    layout="centered",
    page_icon="🎙️"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Mono', monospace !important; background-color: #0a0a0a !important; color: #f0f0f0 !important; }
.stApp { background-color: #0a0a0a; }
h1 { font-family: 'Bebas Neue', sans-serif !important; font-size: 68px !important; letter-spacing: 6px !important; color: #ff3c3c !important; line-height: 1 !important; }
h2 { font-family: 'Bebas Neue', sans-serif !important; font-size: 28px !important; letter-spacing: 4px !important; color: #ff3c3c !important; margin-top: 32px !important; }
h3 { font-family: 'Space Mono', monospace !important; font-size: 11px !important; letter-spacing: 3px !important; color: #ff3c3c !important; text-transform: uppercase; }
.stButton > button { width: 100% !important; background: linear-gradient(135deg, #ff3c3c, #cc1a1a) !important; color: white !important; border: none !important; border-radius: 12px !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 26px !important; letter-spacing: 4px !important; padding: 18px !important; box-shadow: 0 8px 32px rgba(255,60,60,0.3) !important; }
footer { display: none !important; } #MainMenu { display: none !important; } header { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
#  DSP UTILS
# ═══════════════════════════════════════════════════

def frames_fn(x, sr, ms=20, hop_ms=10):
    f = max(1, int(sr * ms / 1000))
    h = max(1, int(sr * hop_ms / 1000))
    n = 1 + max(0, (len(x) - f) // h)
    idx = np.clip(np.arange(f)[None,:] + np.arange(n)[:,None] * h, 0, len(x) - 1)
    return x[idx]

def rms_db_frames(x, sr, ms=20, hop_ms=10):
    return 20 * np.log10(np.sqrt(np.mean(frames_fn(x, sr, ms, hop_ms)**2, axis=1) + EPS))

def rms_act(x, sr):
    d = rms_db_frames(x, sr)
    a = d[d > -42]
    return float(np.percentile(a, 50)) if a.size > 5 else float(np.mean(d))

def peak_db(x):
    return float(20 * np.log10(np.max(np.abs(x)) + EPS))

def crest_factor(x):
    return float(20 * np.log10((np.max(np.abs(x)) + EPS) / (np.sqrt(np.mean(x**2)) + EPS)))

def spectral_bands(x, sr):
    N = min(len(x), 65536)
    X = np.abs(rfft(x[:N] * np.hanning(N)))**2
    freqs = rfftfreq(N, 1.0 / sr)
    total = np.sum(X) + EPS
    bands = {
        'sub':        (20,    80),
        'low':        (80,    200),
        'lo_mid':     (200,   500),
        'mid':        (500,   2000),
        'hi_mid':     (2000,  5000),
        'presence':   (2000,  6000),
        'sib_zone':   (5000,  10000),
        'air':        (6000,  12000),
        'brilliance': (12000, min(20000, sr/2 - 100)),
    }
    return {n: float(10 * np.log10(np.sum(X[(freqs>=f0)&(freqs<f1)]) / total + EPS))
            for n, (f0, f1) in bands.items()}

# ═══════════════════════════════════════════════════
#  MESURES SPÉCIALISÉES VOIX
# ═══════════════════════════════════════════════════

def mesure_bruit_fond(x, sr):
    """Plancher de bruit — percentile 5% des frames"""
    rms_f = rms_db_frames(x, sr, 50, 25)
    return float(np.percentile(rms_f, 5))

def mesure_snr(x, sr):
    """Rapport signal/bruit estimé"""
    rms_f = rms_db_frames(x, sr, 50, 25)
    noise = float(np.percentile(rms_f, 5))
    sig_frames = rms_f[rms_f > -42]
    sig = float(np.percentile(sig_frames, 50)) if sig_frames.size > 5 else -40.0
    return sig - noise, noise, sig

def mesure_clipping(x):
    """Détecte clipping et near-clipping"""
    clips     = int(np.sum(np.abs(x) >= 0.998))
    near      = int(np.sum(np.abs(x) >= 0.95))
    # Clipping numérique : séquences consécutives de samples à max
    consec = 0
    max_consec = 0
    for v in np.abs(x):
        if v >= 0.998:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0
    return clips, near, max_consec

def mesure_plosives(x, sr):
    """
    Plosives = pics d'énergie sub (<120 Hz) très courts et intenses.
    Retourne (count, positions_ms)
    """
    b, a = signal.butter(4, min(120.0/(sr/2), 0.95), btype='lowpass')
    x_lo = signal.lfilter(b, a, x)
    hop = max(1, int(sr * 5 / 1000))
    n_frames = (len(x_lo) - hop) // hop
    frames_lo = np.array([np.max(np.abs(x_lo[i*hop:(i+1)*hop])) for i in range(n_frames)])
    if len(frames_lo) == 0:
        return 0, []
    thr = np.percentile(frames_lo, 97)
    # Une plosive = frame qui dépasse 2x le seuil haut
    mask = frames_lo > thr * 2.0
    # Dédoublonne les détections proches
    positions_ms = []
    last = -500
    for i, m in enumerate(mask):
        if m:
            t_ms = i * 5
            if t_ms - last > 200:
                positions_ms.append(t_ms)
                last = t_ms
    return len(positions_ms), positions_ms

def mesure_sibilance(x, sr):
    """
    Sibilance = énergie 5-10 kHz par rapport à la présence 2-5 kHz.
    Retourne (sib_db, pres_db, ecart_db)
    """
    N = min(len(x), 65536)
    X = np.abs(rfft(x[:N] * np.hanning(N)))**2
    freqs = rfftfreq(N, 1.0/sr)
    total = np.sum(X) + EPS
    sib  = float(10 * np.log10(np.sum(X[(freqs>=5000)&(freqs<10000)]) / total + EPS))
    pres = float(10 * np.log10(np.sum(X[(freqs>=2000)&(freqs< 5000)]) / total + EPS))
    return sib, pres, sib - pres

def mesure_clics_bouche(x, sr):
    """
    Clics de bouche = pics très courts (< 5ms) en haute fréquence (> 3kHz)
    """
    b, a = signal.butter(4, min(3000.0/(sr/2), 0.95), btype='highpass')
    x_hi = signal.lfilter(b, a, x)
    hop = max(1, int(sr * 2 / 1000))
    n_frames = (len(x_hi) - hop) // hop
    if n_frames == 0:
        return 0
    frames_hi = np.array([np.max(np.abs(x_hi[i*hop:(i+1)*hop])) for i in range(n_frames)])
    thr = np.percentile(frames_hi, 98)
    return int(np.sum(frames_hi > thr * 2.5))

def mesure_regularite_volume(x, sr):
    """
    Variance du volume sur les frames actives.
    Un bon compresseur donne < 4 dB d'écart.
    """
    rms_f = rms_db_frames(x, sr, 100, 50)
    active = rms_f[rms_f > -42]
    if active.size < 4:
        return 0.0, 0.0, 0.0
    return float(np.std(active)), float(np.percentile(active, 10)), float(np.percentile(active, 90))

def mesure_dynamique_naturelle(x, sr):
    """
    Crest factor + dynamic range = dynamique naturelle de la voix.
    Une voix non traitée a 18-25 dB de crest.
    """
    cf = crest_factor(x)
    rms_f = rms_db_frames(x, sr, 200, 100)
    active = rms_f[rms_f > -60]
    dr = float(np.percentile(active, 95) - np.percentile(active, 5)) if active.size > 4 else 0.0
    return cf, dr

def mesure_reverb_piece(x, sr):
    """
    Estime la réverb de pièce via le decay moyen après les transitoires.
    Un signal sec a un decay rapide (< -3 dB/frame à 10ms).
    Retourne (rt60_approx_ms, reverb_niveau)
    """
    rms_f = rms_db_frames(x, sr, 10, 5)
    diff = np.diff(rms_f)
    # Decay après les attaques (frames qui chutent après un pic)
    peaks = np.where((rms_f[1:] < rms_f[:-1] - 3) & (rms_f[:-1] > -30))[0]
    if len(peaks) == 0:
        return 0.0, "indéterminable"
    decay_rates = []
    for p in peaks[:20]:
        # Cherche combien de frames pour chuter de 20 dB
        start = rms_f[p]
        for j in range(1, min(200, len(rms_f) - p)):
            if rms_f[p + j] < start - 20:
                decay_rates.append(j * 5)  # en ms
                break
    if not decay_rates:
        return 0.0, "sec"
    rt20 = float(np.median(decay_rates))
    rt60_approx = rt20 * 3  # RT60 ≈ 3 × RT20
    if rt60_approx < 150:    niveau = "très sec — idéal"
    elif rt60_approx < 300:  niveau = "sec — bon"
    elif rt60_approx < 600:  niveau = "léger — acceptable"
    elif rt60_approx < 1200: niveau = "présent — problématique"
    else:                    niveau = "fort — à traiter"
    return rt60_approx, niveau

def mesure_effet_proximite(x, sr):
    """
    Effet de proximité = excès de graves 80-200 Hz par rapport aux lo-mids.
    Si low >> lo_mid → trop près du micro.
    """
    N = min(len(x), 65536)
    X = np.abs(rfft(x[:N] * np.hanning(N)))**2
    freqs = rfftfreq(N, 1.0/sr)
    total = np.sum(X) + EPS
    low   = float(10 * np.log10(np.sum(X[(freqs>=80) &(freqs<200)]) / total + EPS))
    lo_mid= float(10 * np.log10(np.sum(X[(freqs>=200)&(freqs<500)]) / total + EPS))
    sub   = float(10 * np.log10(np.sum(X[(freqs>=20) &(freqs<80) ]) / total + EPS))
    return low - lo_mid, low, lo_mid, sub

def mesure_attaques(x, sr):
    """
    Score d'attaque des mots — netteté des transitoires voix.
    Voix naturelle non compressée : > 5/10
    """
    hop = max(1, int(sr * 2 / 1000))
    env = np.sqrt(np.mean(frames_fn(x, sr, 5, 2)**2, axis=1) + EPS)
    if len(env) < 2:
        return 5.0
    rises = np.diff(env)
    rises = rises[rises > 0]
    if len(rises) == 0:
        return 0.0
    score = float(np.percentile(rises, 90)) / (float(np.mean(env)) + EPS)
    return float(min(score * 3, 10.0))

def mesure_fins_phrases(x, sr):
    """
    Détecte si les fins de phrases disparaissent ou sont coupées.
    Cherche des chutes brutales du niveau actif.
    """
    rms_f = rms_db_frames(x, sr, 50, 25)
    active = rms_f > -38
    if active.sum() < 4:
        return 0, 0
    diff = np.diff(rms_f.astype(float))
    # Chutes brutales = baisses > 15 dB en 1 frame sur parties actives
    brutal_drops = 0
    soft_ends = 0
    for i in range(len(diff)-1):
        if active[i] and diff[i] < -15:
            brutal_drops += 1
        elif active[i] and -15 <= diff[i] < -8:
            soft_ends += 1
    return brutal_drops, soft_ends

def mesure_nasalite(b):
    """
    Nasalité = excès de lo_mid (200-500 Hz) par rapport au mid.
    """
    return b['lo_mid'] - b['mid']

def mesure_boxy(b):
    """
    Boxy / fermé = accumulation à 300-500 Hz.
    """
    return b['lo_mid']

def mesure_agressivite(b):
    """
    Agressif / dur = excès de hi_mid (2-5 kHz).
    """
    return b['hi_mid']

def mesure_sourd(b):
    """
    Sourd / étouffé = manque de hi_mid et air.
    """
    return (b['hi_mid'] + b['air']) / 2

def mesure_minceur(b):
    """
    Fin / maigre = manque de low et lo_mid.
    """
    return (b['low'] + b['lo_mid']) / 2

# ═══════════════════════════════════════════════════
#  ANALYSE COMPLÈTE
# ═══════════════════════════════════════════════════

def analyser_voix(x, sr):
    a = {}
    a['rms']           = rms_act(x, sr)
    a['peak']          = peak_db(x)
    a['crest'], a['dr']= mesure_dynamique_naturelle(x, sr)
    a['clips'], a['near_clips'], a['consec_clips'] = mesure_clipping(x)
    a['snr'], a['noise_floor'], a['sig_level']     = mesure_snr(x, sr)
    a['plosives_count'], a['plosives_pos']         = mesure_plosives(x, sr)
    a['sib'], a['pres_db'], a['sib_ecart']         = mesure_sibilance(x, sr)
    a['clics_bouche']  = mesure_clics_bouche(x, sr)
    a['vol_std'], a['vol_p10'], a['vol_p90']       = mesure_regularite_volume(x, sr)
    a['rt60'], a['reverb_niveau']                  = mesure_reverb_piece(x, sr)
    a['prox_ecart'], a['low_db'], a['lo_mid_db'], a['sub_db'] = mesure_effet_proximite(x, sr)
    a['attaques']      = mesure_attaques(x, sr)
    a['drops'], a['soft_ends'] = mesure_fins_phrases(x, sr)
    a['bands']         = spectral_bands(x, sr)
    b = a['bands']
    a['nasalite']      = mesure_nasalite(b)
    a['boxy']          = mesure_boxy(b)
    a['agressivite']   = mesure_agressivite(b)
    a['sourd']         = mesure_sourd(b)
    a['minceur']       = mesure_minceur(b)
    return a

# ═══════════════════════════════════════════════════
#  DIAGNOSTICS — 6 BLOCS INGÉ SON
# ═══════════════════════════════════════════════════

def bloc_proprete(a):
    """1. PROPRETÉ — bruit, clipping, plosives, clics, réverb"""
    d = []

    # Clipping
    if a['clips'] > 0 and a['consec_clips'] >= 3:
        d.append(("🔴", "Saturation numérique — REFAIRE LA PRISE",
            f"{a['clips']} samples écrêtés dont {a['consec_clips']} consécutifs. C'est de la distorsion irréparable. Baisse le gain d'entrée de 6 dB et refais la prise."))
    elif a['clips'] > 0:
        d.append(("🟡", f"Légère saturation ({a['clips']} samples)",
            "Quelques samples écrêtés. Souvent inaudible si isolé, mais surveille. Baisse légèrement le gain d'entrée à la prochaine prise."))
    elif a['near_clips'] > 50:
        d.append(("🟡", f"Niveau très chaud — risque de saturation ({a['near_clips']} near-clips)",
            "Le signal frôle souvent 0 dBFS. Pas encore de distorsion mais une crête plus forte saturationera. Préfère enregistrer à -6 dBFS max."))
    else:
        d.append(("🟢", "Pas de saturation — prise propre",
            "Aucun écrêtage détecté. Le niveau d'enregistrement est sain."))

    # Bruit de fond
    if a['noise_floor'] > -35:
        d.append(("🔴", f"Bruit de fond fort ({a['noise_floor']:.0f} dB)",
            f"SNR estimé à {a['snr']:.0f} dB seulement. La pièce est bruyante ou le gain trop élevé. Ferme les fenêtres, coupe la clim, éloigne-toi des sources de bruit. Vise un bruit de fond < -50 dB."))
    elif a['noise_floor'] > -45:
        d.append(("🟡", f"Bruit de fond modéré ({a['noise_floor']:.0f} dB / SNR {a['snr']:.0f} dB)",
            "Audible dans les silences. Un plugin de noise reduction léger (RX, Waves NS1) suffira. Essaie d'améliorer l'acoustique de la pièce pour les prochaines sessions."))
    else:
        d.append(("🟢", f"Bruit de fond faible ({a['noise_floor']:.0f} dB / SNR {a['snr']:.0f} dB)",
            "La prise est silencieuse entre les mots. Bonne acoustique ou bon placement de micro."))

    # Réverb de pièce
    if a['rt60'] > 0:
        if a['rt60'] > 1200:
            d.append(("🔴", f"Réverb de pièce forte (RT60 ~{a['rt60']:.0f} ms)",
                "La pièce résonne beaucoup. Ça va coller la voix et empêcher la compression de fonctionner correctement. Enregistre dans un placard, tends des couvertures, ou utilise un filtre acoustique derrière le micro."))
        elif a['rt60'] > 600:
            d.append(("🟡", f"Réverb de pièce présente (RT60 ~{a['rt60']:.0f} ms)",
                "La pièce colore la voix. Audible à l'écoute attentive. Un traitement acoustique DIY (mousse, couvertures) réduirait ça. Traitement possible avec un plugin de déréverbération (RX Dialogue Isolate)."))
        elif a['rt60'] > 300:
            d.append(("🟡", f"Légère réverb de pièce (RT60 ~{a['rt60']:.0f} ms)",
                "Très légère coloration acoustique. Pour la plupart des styles c'est ok. Pour un son ultra-sec, une couverture derrière la tête suffirait."))
        else:
            d.append(("🟢", f"Prise sèche — bonne acoustique (RT60 ~{a['rt60']:.0f} ms)",
                "La pièce ne colore pas la voix. Bon environnement d'enregistrement."))

    # Plosives
    if a['plosives_count'] >= 5:
        d.append(("🔴", f"Plosives fréquentes ({a['plosives_count']} détectées)",
            "Les P/B/T font des \"boum\" graves dans le micro. Un anti-pop est indispensable. Place-le à 5-10 cm du micro. Sinon enregistre légèrement de côté."))
    elif a['plosives_count'] >= 2:
        d.append(("🟡", f"Quelques plosives ({a['plosives_count']} détectées)",
            "Quelques impacts graves détectés. Un filtre passe-haut à 80 Hz peut atténuer ça. Un anti-pop éviterait ça à la prochaine prise."))
    else:
        d.append(("🟢", "Pas de plosives détectées",
            "Aucun impact grave parasité. Bonne technique ou bon anti-pop."))

    # Clics de bouche
    if a['clics_bouche'] >= 8:
        d.append(("🔴", f"Beaucoup de clics de bouche ({a['clics_bouche']} détectés)",
            "La bouche claque, salive excessive ou micro trop proche. Bois de l'eau entre les prises. Du jus de pomme non sucré aide aussi. Recul légèrement du micro. Ces clics sont très difficiles à enlever en post."))
    elif a['clics_bouche'] >= 3:
        d.append(("🟡", f"Quelques clics de bouche ({a['clics_bouche']} détectés)",
            "Quelques petits clics détectés. Écoute attentivement avant de livrer. Un plugin DeClick (RX, iZotope) peut les nettoyer un par un."))
    else:
        d.append(("🟢", "Pas de clics de bouche détectés",
            "Prise propre sur ce point."))

    return d

def bloc_controle(a):
    """2. CONTRÔLE — dynamique, régularité du volume, attaques, fins de phrases"""
    d = []

    # Régularité du volume
    if a['vol_std'] > 8:
        d.append(("🔴", f"Volume très irrégulier (écart {a['vol_std']:.0f} dB)",
            f"La voix monte et descend énormément — de {a['vol_p10']:.0f} dB à {a['vol_p90']:.0f} dB. Technique vocale instable ou trop de mouvement devant le micro. Un compresseur seul ne suffit pas — il faut retravailler la technique ou refaire les passages irréguliers."))
    elif a['vol_std'] > 5:
        d.append(("🟡", f"Volume irrégulier (écart {a['vol_std']:.0f} dB)",
            f"Variations de {a['vol_p10']:.0f} à {a['vol_p90']:.0f} dB. Un compresseur (ratio 3:1 à 4:1) gérera ça mais perdra de la dynamique naturelle. Idéalement, travaille la constance vocale pour moins dépendre du compresseur."))
    else:
        d.append(("🟢", f"Volume régulier (écart {a['vol_std']:.0f} dB)",
            "Bonne constance vocale. Le compresseur n'aura pas besoin de travailler fort — ça préservera la dynamique naturelle."))

    # Dynamique naturelle
    if a['crest'] < 10:
        d.append(("🔴", f"Voix sur-compressée à la source (crest {a['crest']:.0f} dB)",
            "La dynamique est déjà très réduite avant traitement. Est-ce qu'il y a un compresseur actif sur l'interface ou dans le DAW en input ? Désactive-le pour enregistrer. Une voix naturelle a 18-25 dB de crest factor."))
    elif a['crest'] < 14:
        d.append(("🟡", f"Dynamique limitée (crest {a['crest']:.0f} dB)",
            "La voix manque un peu de relief naturel. Vérifie qu'aucun traitement n'est actif à l'enregistrement. Donne-toi la permission de varier l'intensité vocale."))
    elif a['crest'] > 25:
        d.append(("🟡", f"Dynamique très grande (crest {a['crest']:.0f} dB)",
            "Grandes variations d'intensité. C'est naturel mais difficile à contrôler au mix. Un compresseur léger en amont (ratio 2:1, attaque 20ms) stabilisera sans tuer la vie."))
    else:
        d.append(("🟢", f"Dynamique naturelle équilibrée (crest {a['crest']:.0f} dB)",
            "Bonne dynamique vocale. Facile à travailler au mix sans perdre l'émotion."))

    # Attaques des mots
    if a['attaques'] < 3:
        d.append(("🔴", f"Attaques molles ({a['attaques']:.1f}/10)",
            "Les mots démarrent sans énergie — soit sur-compression, soit technique vocale molle. Ça rend la voix floue et peu présente. Ouvre l'attaque du compresseur (> 20ms) ou travaille les consonnes d'attaque."))
    elif a['attaques'] < 5:
        d.append(("🟡", f"Attaques peu marquées ({a['attaques']:.1f}/10)",
            "Les débuts de mots manquent de punch. Un transient shaper (+2 dB sur l'attaque) peut redonner du tranchant. À l'enregistrement, articule davantage les consonnes initiales."))
    else:
        d.append(("🟢", f"Bonnes attaques ({a['attaques']:.1f}/10)",
            "Les mots démarrent avec de l'énergie. La voix est présente et nette."))

    # Fins de phrases
    if a['drops'] >= 4:
        d.append(("🔴", f"Fins de phrases coupées ({a['drops']} détectées)",
            "Des mots ou syllabes finales disparaissent brutalement. Soit la voix baisse trop en fin de phrase, soit le chanteur/rappeur se déplace du micro. Refais ces passages ou utilise l'automation de volume."))
    elif a['drops'] >= 1:
        d.append(("🟡", f"Quelques fins de phrases faibles ({a['drops']} détectées)",
            "Quelques chutes de niveau en fin de phrase. L'automation de volume ou un compresseur montant règleront ça. Garde en tête pour les prochaines prises."))
    else:
        d.append(("🟢", "Fins de phrases tenues",
            "Pas de chute brutale de niveau détectée. Bonne tenue vocale jusqu'au bout des phrases."))

    return d

def bloc_comprehension(a):
    """3. COMPRÉHENSION — sibilance, intelligibilité fréquentielle, effet de proximité"""
    d = []
    b = a['bands']

    # Sibilance
    if a['sib_ecart'] > 3:
        d.append(("🔴", f"Sibilance excessive — S/CH agressifs (écart {a['sib_ecart']:.0f} dB)",
            "Les S, CH, T sifflent et agressent l'oreille. Un de-esser centré à 7-8 kHz est indispensable. Threshold bas, ratio 4:1. Dans BandLab/GarageBand, cherche 'De-esser' ou 'Vocal enhancer'."))
    elif a['sib_ecart'] > 1:
        d.append(("🟡", f"Légère sibilance ({a['sib_ecart']:.0f} dB)",
            "Les sibilantes sont un peu présentes. Un de-esser léger ou un EQ avec une légère coupe à 8 kHz suffira."))
    else:
        d.append(("🟢", "Sibilance sous contrôle",
            "Les S et CH sont bien équilibrés. Pas de de-esser indispensable."))

    # Intelligibilité — présence 2-5 kHz
    if b['hi_mid'] < -28:
        d.append(("🔴", f"Voix peu intelligible — manque de définition (2-5 kHz : {b['hi_mid']:.0f} dB)",
            "La zone d'intelligibilité des consonnes est très faible. On entend la voix mais on comprend mal les mots. Un boost de +3 à +5 dB à 3 kHz avec un EQ en cloche large (Q=0.8) est prioritaire."))
    elif b['hi_mid'] < -22:
        d.append(("🟡", f"Intelligibilité à améliorer (2-5 kHz : {b['hi_mid']:.0f} dB)",
            "Un boost de +2 dB à 3.5 kHz donnera plus de clarté aux consonnes et à l'articulation."))
    else:
        d.append(("🟢", f"Bonne intelligibilité (2-5 kHz : {b['hi_mid']:.0f} dB)",
            "La zone de définition des consonnes est bien représentée. La voix est claire et compréhensible."))

    # Effet de proximité
    if a['prox_ecart'] > 6:
        d.append(("🔴", f"Effet de proximité fort — trop près du micro (écart {a['prox_ecart']:.0f} dB)",
            "Les graves 80-200 Hz dominent énormément. C'est l'effet de proximité cardioïde — tu chantes/rapptes trop près. Recule à 15-20 cm. Un HPF à 120 Hz atténuera aussi ça au mix."))
    elif a['prox_ecart'] > 3:
        d.append(("🟡", f"Légère proximité micro (écart {a['prox_ecart']:.0f} dB)",
            "Un peu trop de graves liés à la distance. Un HPF à 100 Hz et un léger dip à 150 Hz nettoieront ça facilement."))
    else:
        d.append(("🟢", "Bonne distance micro",
            "Pas d'effet de proximité excessif. La distance au micro semble correcte."))

    # Sub parasite
    if a['sub_db'] > -20:
        d.append(("🟡", f"Sub parasite présent (20-80 Hz : {a['sub_db']:.0f} dB)",
            "Bruit de corps, ventilation ou vibration de pièce. Un HPF à 80 Hz est indispensable — ça n'affecte pas la voix mais nettoie le sub inutile."))

    return d

def bloc_couleur(a):
    """4. COULEUR — timbre, bandes fréquentielles, caractère sonore"""
    d = []
    b = a['bands']

    # Grave
    if b['low'] > -8:
        d.append(("🟡", f"Voix très grave / chaleureuse (80-200 Hz : {b['low']:.0f} dB)",
            "Beaucoup de graves dans la voix. Chaleureux mais peut manquer de clarté dans un mix chargé. Sculpte légèrement à 150-180 Hz si elle étouffe les autres éléments."))
    elif b['low'] < -20:
        d.append(("🟡", f"Voix fine, peu de grave (80-200 Hz : {b['low']:.0f} dB)",
            "La fondamentale vocale est faible. La voix peut sonner mince ou nasale. Un léger boost à 150-200 Hz donnera plus de corps."))

    # Nasalité
    if a['nasalite'] > 4:
        d.append(("🔴", f"Voix nasale (lo-mid domine de {a['nasalite']:.0f} dB)",
            "Excès à 200-500 Hz par rapport aux mids — c'est ce qui donne le côté 'dans les narines'. Coupe entre 250-350 Hz avec une cloche étroite (Q=2), -3 à -5 dB. Sweep lentement pour trouver la fréquence exacte."))
    elif a['nasalite'] > 2:
        d.append(("🟡", f"Légère nasalité ({a['nasalite']:.0f} dB)",
            "Légère accumulation à 300 Hz. Une coupe de -2 dB à 280 Hz ouvrira la voix."))

    # Boxy / fermé
    if a['boxy'] > -14:
        d.append(("🔴", f"Voix boxy / fermée (200-500 Hz : {a['boxy']:.0f} dB)",
            "Zone lo-mid surchargée — la voix sonne dans une boîte. C'est souvent la réflexion des murs proches. Coupe à 300-400 Hz (-4 à -6 dB). Éloigne-toi des murs à la prochaine prise."))
    elif a['boxy'] > -18:
        d.append(("🟡", f"Légèrement boxy ({a['boxy']:.0f} dB)",
            "Un peu d'accumulation dans le lo-mid. Une légère coupe à 350 Hz nettoierait le timbre."))

    # Agressivité / dureté
    if a['agressivite'] > -12:
        d.append(("🟡", f"Voix agressive / dure (2-5 kHz : {a['agressivite']:.0f} dB)",
            "Zone de présence très chargée. Peut fatiguer l'oreille sur la durée. Baisse légèrement 3-4 kHz (-2 dB) ou utilise un EQ dynamique déclenché uniquement sur les passages intenses."))

    # Sourd / étouffé
    if a['sourd'] < -26:
        d.append(("🔴", f"Voix sourde / étouffée (hi-mid+air : {a['sourd']:.0f} dB)",
            "La voix manque cruellement d'ouverture et de définition. Boost de +3 à +4 dB à 3.5 kHz et high shelf +2 dB à 10 kHz. Vérifier aussi que le micro n'est pas obstrué ou mal orienté."))
    elif a['sourd'] < -22:
        d.append(("🟡", f"Voix un peu fermée ({a['sourd']:.0f} dB)",
            "Manque d'air et de définition. Un high shelf +2 dB à partir de 8 kHz apporterait de la légèreté."))

    # Fin / maigre
    if a['minceur'] < -22:
        d.append(("🟡", f"Voix fine / maigre (grave+lo-mid : {a['minceur']:.0f} dB)",
            "La voix manque de corps et de chaleur. Un boost de +2 dB à 150-200 Hz donnera plus de présence physique."))

    # Air / brillance
    if b['air'] < -30:
        d.append(("🟡", f"Manque d'air (6-12 kHz : {b['air']:.0f} dB)",
            "Voix un peu terne dans les aigus. Un high shelf +2 dB à partir de 10 kHz apporterait de l'espace et de la légèreté."))
    elif b['air'] > -18:
        d.append(("🟡", f"Voix très aérée/brillante (6-12 kHz : {b['air']:.0f} dB)",
            "Beaucoup d'air — attention à la sibilance et à la fatigue d'écoute. Vérifie que ça ne sonne pas artificiel."))

    # Si timbre globalement bon
    if not any(e == "🔴" for e, _, _ in d):
        d.append(("🟢", "Timbre global équilibré",
            "Aucun défaut de couleur majeur. La voix a un timbre naturel et travaillable."))

    return d

def bloc_problemes(a):
    """5. PROBLÈMES — origine + réaction au traitement"""
    d = []

    # Ce qui vient de la prise
    problemes_prise = []
    if a['clips'] > 0:
        problemes_prise.append("saturation")
    if a['noise_floor'] > -40:
        problemes_prise.append("bruit de fond")
    if a['plosives_count'] >= 2:
        problemes_prise.append("plosives")
    if a['clics_bouche'] >= 3:
        problemes_prise.append("clics de bouche")
    if a['prox_ecart'] > 5:
        problemes_prise.append("effet de proximité")

    if problemes_prise:
        d.append(("🔴", f"Problèmes de PRISE : {', '.join(problemes_prise)}",
            "Ces problèmes viennent de l'enregistrement. Certains sont impossibles à corriger après coup (saturation). Améliore les conditions pour les prochaines prises."))

    # Ce qui vient de la pièce
    problemes_piece = []
    if a['rt60'] > 600:
        problemes_piece.append(f"réverb de pièce (~{a['rt60']:.0f}ms)")
    if a['noise_floor'] > -40:
        problemes_piece.append("bruit ambiant")
    if a['boxy'] > -15:
        problemes_piece.append("coloration boxy")

    if problemes_piece:
        d.append(("🟡", f"Problèmes de PIÈCE : {', '.join(problemes_piece)}",
            "Ces problèmes viennent de l'acoustique. Traitement possible mais limité. Un plugin de déréverbération (RX Dialogue Isolate) aide, mais un bon traitement acoustique à la source reste la meilleure solution."))

    # Ce qui vient de la performance
    problemes_perf = []
    if a['vol_std'] > 6:
        problemes_perf.append("volume irrégulier")
    if a['drops'] >= 3:
        problemes_perf.append("fins de phrases coupées")
    if a['attaques'] < 4:
        problemes_perf.append("attaques molles")

    if problemes_perf:
        d.append(("🟡", f"Problèmes de PERFORMANCE : {', '.join(problemes_perf)}",
            "Ces problèmes viennent de la technique vocale ou du placement micro. Partiellement corrigeables au mix (automation, compression) mais mieux vaut refaire les passages problématiques."))

    # Ce qui réagira bien au traitement
    bien = []
    if -22 <= a['rms'] <= -10:
        bien.append("niveau adapté au traitement")
    if a['snr'] > 30:
        bien.append("bon SNR — compression propre")
    if a['crest'] >= 14:
        bien.append("dynamique naturelle — compression efficace")
    if a['rt60'] < 400:
        bien.append("prise sèche — EQ précis")
    if a['clips'] == 0:
        bien.append("pas de saturation — traitement sans artefact")

    if bien:
        d.append(("🟢", f"Réagira BIEN au traitement : {', '.join(bien)}",
            "Ces caractéristiques garantissent que le traitement DSP donnera des résultats propres et prévisibles."))

    # Ce qui réagira mal
    mal = []
    if a['rt60'] > 600:
        mal.append(f"réverb de pièce — l'EQ va l'amplifier")
    if a['noise_floor'] > -40:
        mal.append("bruit de fond — la compression va le pomper")
    if a['clips'] > 0 and a['consec_clips'] >= 3:
        mal.append("saturation — la distorsion est gravée dans le signal")
    if a['vol_std'] > 8:
        mal.append("dynamique excessive — compression agressive nécessaire")

    if mal:
        d.append(("🔴", f"Réagira MAL au traitement : {', '.join(mal)}",
            "Ces éléments limitent ce qu'on peut faire en post. La compression va pomper le bruit, l'EQ va amplifier la réverb. La meilleure correction reste une nouvelle prise dans de meilleures conditions."))

    return d

def bloc_verdict(a):
    """6. VERDICT INGÉ — ordre de priorité"""
    d = []

    # Compte les problèmes rouges
    tous = bloc_proprete(a) + bloc_controle(a) + bloc_comprehension(a) + bloc_couleur(a)
    rouges = [x for x in tous if x[0] == "🔴"]
    jaunes  = [x for x in tous if x[0] == "🟡"]

    # Verdict global
    if len(rouges) >= 4:
        d.append(("🔴", "REFAIRE LA PRISE — trop de problèmes majeurs",
            f"{len(rouges)} problèmes critiques détectés. Le traitement ne pourra pas tout sauver. Une nouvelle prise dans de meilleures conditions est la meilleure décision."))
    elif len(rouges) >= 2:
        d.append(("🟡", f"Prise utilisable avec corrections ({len(rouges)} problèmes critiques, {len(jaunes)} à surveiller)",
            "La prise peut être sauvée mais nécessite un traitement soigné. Traite les 🔴 en priorité avant de passer aux ajustements fins."))
    elif len(rouges) == 1:
        d.append(("🟡", f"Bonne prise avec 1 point critique à corriger",
            "Quasi-prête. Règle le problème critique puis le traitement sera simple et propre."))
    else:
        d.append(("🟢", "Bonne prise — prête pour le traitement",
            f"Aucun problème critique. {len(jaunes)} points à affiner mais la prise est exploitable directement. Le traitement sera propre et efficace."))

    # Ordre de priorité
    priorites = []
    if a['clips'] > 0 and a['consec_clips'] >= 3:
        priorites.append("1. REFAIRE la prise (saturation irréparable)")
    if a['noise_floor'] > -35:
        priorites.append("2. Traiter le bruit de fond (RX ou Noise Gate)")
    if a['plosives_count'] >= 3:
        priorites.append("3. Corriger les plosives (HPF 80 Hz)")
    if a['rt60'] > 600:
        priorites.append("4. Traiter la réverb (déréverbération)")
    if a['vol_std'] > 6:
        priorites.append("5. Compression (régulariser le volume)")
    if a['boxy'] > -16 or a['nasalite'] > 3:
        priorites.append("6. EQ correctif (boue/nasalité)")
    if a['sib_ecart'] > 2:
        priorites.append("7. De-esser (sibilance)")
    if a['sourd'] < -24:
        priorites.append("8. EQ présence/air (ouvrir le timbre)")

    if priorites:
        d.append(("🔵", "Ordre de traitement recommandé",
            " → ".join(priorites)))

    # Ce qu'il faut évaluer à l'oreille
    d.append(("👂", "À évaluer À L'OREILLE (non mesurable)",
        "Justesse (pitch) · Timing musical · Émotion · Interprétation · Intelligibilité des mots · Cohérence entre prises · Qualité de la performance"))

    return d

# ═══════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════

st.markdown("<h1>🎙️ VOIXANALYZER</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#555;font-size:11px;letter-spacing:3px;margin-top:-10px'>DIAGNOSTIC PRO D'UNE PRISE VOIX — GRATUIT</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)

st.markdown("<h3>📂 Ta prise voix brute</h3>", unsafe_allow_html=True)
st.markdown("<p style='color:#444;font-size:11px'>Charge ta prise non traitée — WAV de préférence, brut de micro</p>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "WAV · MP3 · FLAC · OGG · M4A",
    type=["wav","mp3","flac","ogg","m4a"],
    label_visibility="visible"
)

if uploaded is None:
    st.info("👆 Charge ta prise voix pour commencer l'analyse.")
    st.stop()

# Chargement
with st.spinner("Chargement du fichier..."):
    try:
        import librosa
        suffix = "." + uploaded.name.rsplit(".", 1)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        x, sr = librosa.load(tmp_path, sr=44100, mono=True)
        x = (x - float(np.mean(x))).astype(np.float32)
        os.unlink(tmp_path)
        duree = len(x) / sr
    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier : {e}")
        st.stop()

if st.button("🎙️ ANALYSER MA VOIX"):

    with st.spinner("Analyse en cours — mesures DSP..."):
        prog = st.progress(0)
        prog.progress(10, text="🔍 Mesure du bruit de fond...")
        prog.progress(25, text="⚡ Détection saturation / plosives...")
        prog.progress(40, text="🎚️ Analyse spectrale complète...")
        prog.progress(55, text="🗜️ Dynamique et régularité...")
        prog.progress(70, text="🏠 Réverb de pièce...")
        prog.progress(85, text="👂 Sibilance et intelligibilité...")
        a = analyser_voix(x, sr)
        prog.progress(100, text="✅ Analyse terminée !")

    st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)

    # Métriques rapides
    st.markdown("<h3>📊 Mesures clés</h3>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("RMS actif",   f"{a['rms']:.0f} dB")
    c2.metric("Peak",        f"{a['peak']:.1f} dB")
    c3.metric("Crest",       f"{a['crest']:.0f} dB")
    c4.metric("SNR ~",       f"{a['snr']:.0f} dB")
    c5.metric("RT60 ~",      f"{a['rt60']:.0f} ms" if a['rt60'] > 0 else "sec")
    c6.metric("Durée",       f"{duree:.0f}s")

    def render_bloc(titre, items, couleur_titre="#ff3c3c"):
        st.markdown(f"<h2 style='color:{couleur_titre}'>{titre}</h2>", unsafe_allow_html=True)
        for emoji, titre_item, detail in items:
            if emoji == "🔵":
                couleur = "#4499ff"
            elif emoji == "👂":
                couleur = "#aa88ff"
            else:
                couleur = {"🔴":"#ff3c3c","🟡":"#ff8c00","🟢":"#00ff88"}.get(emoji, "#888")
            st.markdown(f"""
            <div style='background:#111;border-left:3px solid {couleur};border-radius:8px;padding:12px 16px;margin:6px 0'>
                <div style='color:{couleur};font-size:13px;font-weight:bold'>{emoji} {titre_item}</div>
                <div style='color:#666;font-size:11px;margin-top:5px;line-height:1.8'>{detail}</div>
            </div>""", unsafe_allow_html=True)

    render_bloc("1 — PROPRETÉ",     bloc_proprete(a))
    render_bloc("2 — CONTRÔLE",     bloc_controle(a))
    render_bloc("3 — COMPRÉHENSION",bloc_comprehension(a))
    render_bloc("4 — COULEUR",      bloc_couleur(a))
    render_bloc("5 — PROBLÈMES",    bloc_problemes(a))
    render_bloc("6 — VERDICT",      bloc_verdict(a), couleur_titre="#00ff88")

    st.markdown("<hr style='border-color:#1a1a1a;margin:24px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#111;border:1px solid #1a1a1a;border-radius:12px;padding:20px;font-size:11px;color:#555;line-height:2'>
    💡 <b style='color:#888'>Pour une meilleure prise la prochaine fois :</b><br>
    📍 15-20 cm du micro — ni trop près (graves) ni trop loin (réverb)<br>
    🔇 Ferme les fenêtres, coupe la clim, décroche le frigo<br>
    👕 Enregistre dans un placard plein de vêtements ou sous une couverture<br>
    🎤 Anti-pop obligatoire — légèrement de côté si tu manques de consonance<br>
    💧 Bois de l'eau (ou jus de pomme) avant pour éviter les clics de bouche<br>
    📱 Aucun traitement à l'entrée — enregistre le signal brut
    </div>
    """, unsafe_allow_html=True)
