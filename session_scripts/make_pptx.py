from pptx import Presentation
from pptx.util import Inches as I, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

NAVY=RGBColor(0x0E,0x2A,0x47); NAVY2=RGBColor(0x13,0x35,0x5A)
GOLD=RGBColor(0xC9,0xA2,0x4B); GOLDL=RGBColor(0xE7,0xC8,0x77)
TEAL=RGBColor(0x1E,0x8F,0x8F); INK=RGBColor(0x1A,0x22,0x2E)
MUT=RGBColor(0x68,0x75,0x8A); WHITE=RGBColor(0xFF,0xFF,0xFF)
PANEL=RGBColor(0xF5,0xF8,0xFC); LINE=RGBColor(0xE4,0xE9,0xF0)
LT=RGBColor(0xCD,0xDD,0xF0)
FONT="Segoe UI"; W=13.333; H=7.5

prs=Presentation(); prs.slide_width=I(W); prs.slide_height=I(H)
blank=prs.slide_layouts[6]

def slide(): return prs.slides.add_slide(blank)
def rect(s,l,t,w,h,fill,line=None,shape=MSO_SHAPE.RECTANGLE,lw=None):
    sp=s.shapes.add_shape(shape,I(l),I(t),I(w),I(h))
    sp.fill.solid(); sp.fill.fore_color.rgb=fill
    if line is None: sp.line.fill.background()
    else: sp.line.color.rgb=line; sp.line.width=Pt(lw or 1)
    sp.shadow.inherit=False
    return sp
def txt(s,l,t,w,h,runs,align=PP_ALIGN.LEFT,anchor=MSO_ANCHOR.TOP,sp_after=4):
    tb=s.shapes.add_textbox(I(l),I(t),I(w),I(h)); tf=tb.text_frame; tf.word_wrap=True
    tf.vertical_anchor=anchor
    if isinstance(runs[0],tuple): runs=[runs]
    for pi,para in enumerate(runs):
        p=tf.paragraphs[0] if pi==0 else tf.add_paragraph()
        p.alignment=align; p.space_after=Pt(sp_after); p.space_before=Pt(0)
        for (t_,sz,col,bold,*rest) in para:
            r=p.add_run(); r.text=t_; f=r.font; f.name=FONT; f.size=Pt(sz)
            f.color.rgb=col; f.bold=bold
            if rest and rest[0]: f.italic=True
    return tb
def chrome(s,kicker,title):
    rect(s,0,0,W,0.07,NAVY); rect(s,W*0.62,0,W*0.38,0.07,GOLD)
    txt(s,0.66,0.5,11,0.4,[[(kicker,12.5,GOLD,True)]])
    txt(s,0.66,0.86,12,1.0,[[(title,30,NAVY,True)]])
def footer(s,right):
    rect(s,0.66,7.02,12.0,0.012,LINE)
    txt(s,0.66,7.06,6,0.3,[[("TAF",10,NAVY,True),("  ·  Toxic Account Forecasting",10,MUT,False)]])
    txt(s,7.0,7.06,5.66,0.3,[[(right,10,MUT,False)]],align=PP_ALIGN.RIGHT)
def stat(s,l,t,w,big,lbl,accent=GOLD,bigcol=NAVY,bigsz=30):
    rect(s,l,t,w,1.25,PANEL,LINE,MSO_SHAPE.ROUNDED_RECTANGLE)
    rect(s,l,t,0.06,1.25,accent)
    txt(s,l+0.22,t+0.16,w-0.4,0.6,[[(big,bigsz,bigcol,True)]])
    txt(s,l+0.22,t+0.74,w-0.4,0.4,[[(lbl,11.5,MUT,False)]])

# ---------- 1 COVER ----------
s=slide(); rect(s,0,0,W,H,NAVY); rect(s,0,0,W,0.09,GOLD)
txt(s,0.66,0.5,11,0.4,[[("ZEAL GROUP   ·   RISK & INTELLIGENCE",14,LT,True)]])
txt(s,0.66,2.05,11,0.5,[[("ALIGNMENT PROPOSAL · OFFICE OF THE CEO",13,GOLDL,True)]])
txt(s,0.6,2.5,12,1.1,[[("AntiFraud Solution",52,WHITE,True)]])
txt(s,0.62,3.62,12,0.7,[[("AI-Driven Intelligence",30,GOLDL,True)]])
txt(s,0.66,4.6,11,1.2,[[("Predict, five days in advance, which accounts will cost the B-book money — and act before they do. A configurable, explainable, self-correcting framework (the TAF engine).",18,LT,False)]])
rect(s,0.66,6.7,12.0,0.012,NAVY2)
txt(s,0.66,6.78,6,0.3,[[("Prepared for the Office of the CEO",13,LT,False)]])
txt(s,7.0,6.78,5.66,0.3,[[("September 2026 · Confidential",13,LT,False)]],align=PP_ALIGN.RIGHT)

# ---------- 2 THE BRIEF, ANSWERED (three points) ----------
s=slide(); chrome(s,"THE BRIEF, ANSWERED — THREE QUESTIONS, MADE SMART","Everything the CEO asked for, on one page.")
pts=[("1","Core metrics & the target",
      "Metric = Abuse-USD caught in advance — money Toxic/Arbitrage accounts extract, flagged before they take it. Baseline measured: $454,136/day at stake (perfect-oracle upper bound). Target: detect ≥ 75% out-of-sample.",
      "Specific metric · Measurable in USD & AUC · Target set"),
     ("2","The logic of the system",
      "Observe → Classify → Predict → Act → Learn. Transparent rules place each account in measurable classes; an explainable model per class predicts who turns to Abuse in 5 days; the class picks the action. Every flag traces to a metric & threshold — no black box.",
      "Explainable end-to-end — the logic can be walked through, step by step"),
     ("3","The result in 2–3 weeks, vs the target",
      "A 2-week live pilot, predicted-vs-actual audited daily against ground truth. At target it recovers ≥ $341k/day (~$86M/yr) of B-book P&L. Clear go / no-go.",
      "Achievable (models already 0.85 AUC) · Relevant (core P&L) · Time-bound (2 weeks)")]
y=1.98
for num,title,body,smart in pts:
    rect(s,0.66,y,0.7,1.42,NAVY,None,MSO_SHAPE.ROUNDED_RECTANGLE)
    txt(s,0.66,y+0.42,0.7,0.6,[[(num,30,GOLDL,True)]],align=PP_ALIGN.CENTER)
    rect(s,1.5,y,11.14,1.42,PANEL,LINE,MSO_SHAPE.ROUNDED_RECTANGLE)
    rect(s,1.5,y,0.06,1.42,GOLD)
    txt(s,1.72,y+0.11,10.7,0.4,[[(title,17,NAVY,True)]])
    txt(s,1.72,y+0.5,10.8,0.7,[[(body,13,INK,False)]])
    txt(s,1.72,y+1.12,10.8,0.3,[[(smart,11.5,MUT,False)]])
    y+=1.6
footer(s,"Each point is expanded in the pages that follow")

# ---------- 3 PROBLEM QUANTIFIED ----------
s=slide(); chrome(s,"THE PROBLEM, QUANTIFIED","The B-book leak has a precise size — and a precise source.")
stat(s,0.66,2.1,2.9,"$454.1k","Addressable abuse / day (perfect oracle)")
stat(s,3.72,2.1,2.9,"~$3.0M","Per week · ~$114M annualised")
stat(s,0.66,3.5,2.9,"98%","of client winnings flow to Toxic/Arbitrage accounts",TEAL,TEAL)
stat(s,3.72,3.5,2.9,"15,598","accounts abusive in ≥1 week (of 36,005)",TEAL,TEAL)
rect(s,6.9,2.1,5.74,2.8,PANEL,LINE,MSO_SHAPE.ROUNDED_RECTANGLE)
txt(s,7.2,2.35,5.2,2.4,[
    [("The accounts that win are almost exactly the accounts that are toxic or arbitraging by definition.",18,NAVY,True)],
    [("",6,NAVY,False)],
    [("This is not a coincidence — it is the thesis. If we can identify who wins next week, we have identified the cost, in advance, with time to act.",15,INK,False)]])
txt(s,0.66,5.1,12,0.8,[[("Cost basis: abuse is a WEEKLY state. The upper bound is a perfect oracle that A-books an account exactly for the weeks it is about to be profitably toxic — the sum of every profitable Toxic/Arbitrage week (cent/contract-normalised USD).",13.5,MUT,False)]])
footer(s,"Source: Zeal account-day corpus, 29 May–4 Sep 2026")

# ---------- 4 UNIVERSE ----------
s=slide(); chrome(s,"FROM FIRST PRINCIPLES — 1 OF 2","Start with the whole universe, and split it honestly.")
rect(s,0.9,2.4,2.4,2.4,RGBColor(0xEA,0xF0,0xF7),NAVY,MSO_SHAPE.OVAL,2)
rect(s,2.5,2.4,2.4,2.4,RGBColor(0xF3,0xEC,0xD8),GOLD,MSO_SHAPE.OVAL,2)
rect(s,3.1,3.0,1.2,1.2,NAVY,WHITE,MSO_SHAPE.OVAL,2)
txt(s,0.95,3.3,1.5,0.8,[[("B — B-Book",14,NAVY,True)]],align=PP_ALIGN.CENTER)
txt(s,3.6,2.6,1.4,0.7,[[("C",15,RGBColor(0x8a,0x6d,0x1f),True)]],align=PP_ALIGN.CENTER)
txt(s,3.35,3.35,0.75,0.6,[[("A",13,WHITE,True)]],align=PP_ALIGN.CENTER)
bl=6.0
txt(s,bl,2.4,6.6,3.0,[
    [("U = B ∪ C.",16,NAVY,True),(" Every account, every day, is one or the other.",16,INK,False)],
    [("C (Toxic / Arbitrage)",16,NAVY,True),(" — accounts whose behaviour matches a configurable, measurable class.",16,INK,False)],
    [("B",16,NAVY,True),(" — everyone else, correctly kept in the B-book: not costing us.",16,INK,False)],
    [("A ⊂ C",16,NAVY,True),(" — the subset that actually costs money (defined next).",16,INK,False)]],sp_after=10)
txt(s,bl,5.6,6.6,0.9,[[("Class definitions are inputs, not hard-coded. Any operator can add, edit or re-weight them.",14,MUT,False)]])
footer(s,"Zeal Group · Confidential")

# ---------- 5 DECISIVE MOVE ----------
s=slide(); chrome(s,"FROM FIRST PRINCIPLES — 2 OF 2","Abuse = Toxic behaviour ∩ money made in the next n days.")
txt(s,0.66,1.95,12,0.8,[[("Being in C is a behaviour. It becomes a cost only when it makes money at our expense. So we intersect C with future profitability:",19,INK,False)]])
rect(s,0.66,2.95,11.98,1.15,NAVY,None,MSO_SHAPE.ROUNDED_RECTANGLE)
txt(s,0.9,3.1,11.5,0.9,[[("A = C ∩ { account gains equity over the next n = 5 days }",24,GOLDL,True)]],align=PP_ALIGN.CENTER,anchor=MSO_ANCHOR.MIDDLE)
txt(s,0.66,4.45,5.9,2.2,[
    [("Why this is the right definition",16,NAVY,True)],
    [("We act only on A, never all of C — harmless 'toxic-looking' flow stays in the B-book earning revenue. False positives are expensive too.",15,INK,False)],
    [("Cost = the withdrawable gain over the window, in normalised USD.",15,INK,False)]],sp_after=9)
txt(s,6.9,4.45,5.7,2.2,[
    [("Why balance & deposits don't enter",16,NAVY,True)],
    [("A client could withdraw the gain at end of day 5. Only money made after the qualifying behaviour is the adverse cost — a clean equity delta, consistent across cent accounts and contract specs.",15,INK,False)]],sp_after=9)
footer(s,"Zeal Group · Confidential")

# ---------- 6 CLASSES & COST (table) ----------
s=slide(); chrome(s,"CORE METRIC #1 — COST BY CLASS · TACKLED IN IMPACT ORDER","Six classes today, ranked by the money at stake.")
data=[("Persistent edge","12,853","46,383","$384,657","0.891"),
      ("High magnitude","3,996","8,901","$374,660","0.865"),
      ("Scalper","10,517","32,111","$292,688","0.751"),
      ("High-exposure / recovery","781","1,511","$179,337","0.952"),
      ("News / volatility","4,829","13,317","$89,210","0.843"),
      ("Martingale","5,125","10,970","$82,230","0.796")]
tb=s.shapes.add_table(7,5,I(0.66),I(2.05),I(11.98),I(3.4)).table
for i,w in enumerate((4.3,2.1,1.9,2.1,1.58)): tb.columns[i].width=I(w)
hdr=["Toxic / Arbitrage class","Abuse accounts (A)","Abuse-weeks","Cost / day (USD)","Model AUC"]
for c,h in enumerate(hdr):
    cell=tb.cell(0,c); cell.text=h; p=cell.text_frame.paragraphs[0]
    p.runs[0].font.size=Pt(12); p.runs[0].font.bold=True; p.runs[0].font.color.rgb=WHITE; p.runs[0].font.name=FONT
    cell.fill.solid(); cell.fill.fore_color.rgb=NAVY
    if c>0: p.alignment=PP_ALIGN.RIGHT
for r,row in enumerate(data,1):
    for c,val in enumerate(row):
        cell=tb.cell(r,c); cell.text=val; p=cell.text_frame.paragraphs[0]
        rn=p.runs[0]; rn.font.size=Pt(13); rn.font.name=FONT
        rn.font.color.rgb=NAVY if c==0 else INK; rn.font.bold=(c==0 or c==4)
        cell.fill.solid(); cell.fill.fore_color.rgb=WHITE if r%2 else PANEL
        if c>0: p.alignment=PP_ALIGN.RIGHT
txt(s,0.66,5.7,12,0.9,[[("Each class is a measurable rule the desk can edit; new classes are added the same way. We attack the highest-impact classes first — every class carries a different share of members that turn into Abuse.",14.5,INK,False)]])
footer(s,"Per-class costs overlap; universe total is de-duplicated ($454.1k/day)")

# ---------- 7 LOGIC UNDER THE HOOD (5 steps) ----------
s=slide(); chrome(s,"CORE REQUIREMENT #2 — THE LOGIC UNDER THE HOOD","Five steps, every one explainable to a first-year analyst.")
steps=[("1 · Observe","Every trade + tick, all servers (sub-2s). Behavioural metrics: markout (adverse selection), hold-time, martingale rate, notional, overnight share."),
       ("2 · Classify","Transparent rules place each account-day in its classes. Every flag traces to a metric and a threshold. No black box."),
       ("3 · Predict","An explainable model per class (random forest) turns today's behaviour into P(becomes Abuse within 5 days) — with feature importances."),
       ("4 · Act","The predicted class picks the response: A-book, widen spread, restrict, or monitor — sized to the expected cost."),
       ("5 · Learn","Tomorrow's ground truth scores today's prediction. Models re-fit walk-forward with transfer learning — a closed loop.")]
x=0.66; wcard=2.28; gap=0.06
for i,(h,b) in enumerate(steps):
    rect(s,x,2.2,wcard,2.5,PANEL,LINE,MSO_SHAPE.ROUNDED_RECTANGLE)
    txt(s,x+0.16,2.36,wcard-0.3,0.4,[[(h,14.5,NAVY,True)]])
    txt(s,x+0.16,2.8,wcard-0.3,1.8,[[(b,11.5,INK,False)]])
    if i<4: txt(s,x+wcard-0.02,3.15,0.4,0.5,[[("→",22,GOLD,True)]])
    x+=wcard+gap+0.2
rect(s,0.66,5.05,11.98,1.4,NAVY,None,MSO_SHAPE.ROUNDED_RECTANGLE)
txt(s,1.0,5.22,11.3,1.1,[[("The intelligence is powerful because it is simple: a chain of measurable definitions, a transparent classifier, and a model whose every driver can be named. If it flags an account, we can always say exactly why.",16.5,LT,False)]],anchor=MSO_ANCHOR.MIDDLE)
footer(s,"Zeal Group · Confidential")

# ---------- 8 MODELS WORK ----------
s=slide(); chrome(s,"THE MODELS ALREADY WORK","Foresight, measured out-of-sample.")
mods=[("High-exposure / recovery","0.952"),("Persistent edge","0.891"),("High magnitude","0.865"),
      ("News / volatility","0.843"),("Martingale","0.796"),("Scalper","0.751")]
tb=s.shapes.add_table(7,2,I(0.66),I(2.1),I(6.2),I(3.4)).table
tb.columns[0].width=I(4.5); tb.columns[1].width=I(1.7)
for c,h in enumerate(("Class model","Holdout AUC")):
    cell=tb.cell(0,c); cell.text=h; rn=cell.text_frame.paragraphs[0].runs[0]
    rn.font.size=Pt(12); rn.font.bold=True; rn.font.color.rgb=WHITE; rn.font.name=FONT
    cell.fill.solid(); cell.fill.fore_color.rgb=NAVY
    if c: cell.text_frame.paragraphs[0].alignment=PP_ALIGN.RIGHT
for r,(m,a) in enumerate(mods,1):
    for c,val in enumerate((m,a)):
        cell=tb.cell(r,c); cell.text=val; rn=cell.text_frame.paragraphs[0].runs[0]
        rn.font.size=Pt(13); rn.font.name=FONT; rn.font.color.rgb=NAVY if c==0 else INK; rn.font.bold=(c==1)
        cell.fill.solid(); cell.fill.fore_color.rgb=WHITE if r%2 else PANEL
        if c: cell.text_frame.paragraphs[0].alignment=PP_ALIGN.RIGHT
stat(s,7.3,2.1,5.3,"0.85","Mean out-of-sample AUC across classes",TEAL,TEAL,34)
txt(s,7.3,3.55,5.3,1.6,[[("AUC is the probability the model ranks a true future-Abuser above a non-Abuser. 0.85 means it is right ~85% of the time at telling tomorrow's cost from tomorrow's noise — today.",16,INK,False)]])
txt(s,7.3,5.15,5.3,0.8,[[("Every model is a random forest: inspectable, reproducible, and it names its own top drivers. No opaque deep-net.",14,MUT,False)]])
footer(s,"Time-ordered holdout, per-class walk-forward")

# ---------- 9 SMART TARGET ----------
s=slide(); chrome(s,"CORE REQUIREMENT #3 — THE RESULT, AGAINST A SMART TARGET","One objective, made SMART.")
rect(s,0.66,1.95,11.98,1.35,NAVY,None,MSO_SHAPE.ROUNDED_RECTANGLE)
txt(s,1.0,2.1,11.3,1.1,[[("Within two weeks, detect ≥ 75% of next-week Abuse-USD in advance (out-of-sample) across the top-six classes, recovering ≥ $341k/day (~$86M/yr) of B-book P&L — measured daily against realised ground truth.",18,LT,False)]],anchor=MSO_ANCHOR.MIDDLE)
smart=[("Specific","Detect the accounts that will become Abuse (A) within 5 days, per class, and act."),
       ("Measurable","OOS detection of Abuse-USD (recall) + precision; USD recovered/day vs the walk-forward baseline."),
       ("Achievable","Models already average 0.85 AUC — comfortably above the 0.75 bar — on Zeal's own history."),
       ("Relevant","Directly protects the B-book's core P&L: $454k/day is at stake today."),
       ("Time-bound","A 2-week live test with a clear go / no-go, reported daily.")]
y=3.6
for k,v in smart:
    rect(s,0.66,y,2.1,0.62,PANEL,LINE,MSO_SHAPE.ROUNDED_RECTANGLE)
    txt(s,0.66,y+0.13,2.1,0.4,[[(k,14,NAVY,True)]],align=PP_ALIGN.CENTER)
    txt(s,3.0,y+0.08,9.6,0.6,[[(v,15,INK,False)]],anchor=MSO_ANCHOR.MIDDLE)
    y+=0.68
footer(s,"Recoverable = target detection rate × measured daily abuse cost")

# ---------- 10 FEEDBACK LOOP ----------
s=slide(); chrome(s,"A SYSTEM THAT GRADES ITSELF","Predicted vs actual, every day — a closed loop.")
loop=[("Predict (day D)","Score every non-member: P(becomes Abuse by D+5). Publish watchlist & actions."),
      ("Observe (D+5)","New ground truth: who actually became Abuse, and the USD they took."),
      ("Score","Precision, recall & USD-captured land on a live dashboard — target audited continuously."),
      ("Re-fit","Walk-forward transfer learning folds the truth back in. Accuracy compounds.")]
x=0.66; wc=2.85
for i,(h,b) in enumerate(loop):
    rect(s,x,2.3,wc,2.3,PANEL,LINE,MSO_SHAPE.ROUNDED_RECTANGLE)
    txt(s,x+0.18,2.46,wc-0.34,0.4,[[(h,14.5,NAVY,True)]])
    txt(s,x+0.18,2.9,wc-0.34,1.6,[[(b,12.5,INK,False)]])
    if i<3: txt(s,x+wc-0.02,3.1,0.4,0.5,[[("↻",22,GOLD,True)]])
    x+=wc+0.18
stat(s,0.66,5.0,5.9,"Self-correcting","Definitions stay fixed; the models adapt to drift automatically.",GOLD,NAVY,22)
stat(s,6.74,5.0,5.9,"Fully auditable","Every prediction is later checked against what really happened.",GOLD,NAVY,22)
footer(s,"Zeal Group · Confidential")

# ---------- 11 AUTO-DISCOVERY ----------
s=slide(); chrome(s,"THE FRONTIER — DISCOVERING ABUSE WE HAVEN'T NAMED YET","Turn the lens around: find the profit we can't yet explain.")
txt(s,0.66,2.15,6.3,3.6,[
    [("Search for accounts that are profitable over the 5-day window and predictable in advance — yet fit no existing class.",16,INK,False)],
    [("These are candidate new abuse classes, surfaced automatically by mining every source: trades, ticks, funding, markout, exposure.",16,INK,False)],
    [("If the AI can explain the pattern → we name a new class and add it.",16,INK,False)],
    [("If it cannot yet → we can still act on the evidence before the explanation matures, when the logic and the money justify it.",16,INK,False)]],sp_after=12)
rect(s,7.3,2.3,5.3,3.0,NAVY,None,MSO_SHAPE.ROUNDED_RECTANGLE)
txt(s,7.6,2.55,4.7,2.6,[[("The framework doesn't just police the abuse we know. It is a discovery engine for the abuse we don't — the difference between reacting to last quarter's arbitrage and getting ahead of next quarter's.",17,LT,False)]],anchor=MSO_ANCHOR.MIDDLE)
footer(s,"Zeal Group · Confidential")

# ---------- 12 SAAS ----------
s=slide(); chrome(s,"BUILT AS A FRAMEWORK, NOT A FIXED MODEL","Vendor-agnostic by design — configure, then run.")
cfg=[("1 · Define universe","Agree the Toxic/Arbitrage classes and metrics. Add your own."),
     ("2 · Set n & cost","Choose the horizon and cost basis. The system optimises the rest."),
     ("3 · Derive settings","Thresholds, actions and priorities fit to your universe automatically."),
     ("4 · Run & test","Watchlists, actions, live dashboard — plus ad-hoc tests for new classes.")]
x=0.66; wc=2.85
for i,(h,b) in enumerate(cfg):
    rect(s,x,2.25,wc,2.1,PANEL,LINE,MSO_SHAPE.ROUNDED_RECTANGLE)
    txt(s,x+0.18,2.4,wc-0.34,0.4,[[(h,14,NAVY,True)]])
    txt(s,x+0.18,2.82,wc-0.34,1.5,[[(b,12.5,INK,False)]])
    if i<3: txt(s,x+wc-0.02,3.0,0.4,0.5,[[("→",22,GOLD,True)]])
    x+=wc+0.18
stat(s,0.66,4.75,3.9,"Any operator","Different definitions plug into the same sound framework.",TEAL,NAVY,20)
stat(s,4.72,4.75,3.9,"Immediate","A short wizard, then results — no bespoke build per client.",TEAL,NAVY,20)
stat(s,8.78,4.75,3.9,"Ad-hoc","Test a new abuse class or a definition change on demand.",TEAL,NAVY,20)
footer(s,"Zeal Group · Confidential")

# ---------- 13 THE ASK ----------
s=slide(); rect(s,0,0,W,H,NAVY); rect(s,0,0,W,0.09,GOLD)
txt(s,0.66,0.55,11,0.4,[[("THE ASK",13,GOLDL,True)]])
txt(s,0.62,1.0,12,1.2,[[("A two-week live pilot on the top-six classes.",38,WHITE,True)]])
cards=[("TARGET","≥ 75% detection","of next-5-day Abuse-USD, out-of-sample"),
       ("PRIZE","~$86M / year","B-book P&L recovered at target"),
       ("PROOF","Daily, audited","predicted vs actual, clear go / no-go")]
x=0.66; wc=3.9
for k,big,sub in cards:
    rect(s,x,2.6,wc,1.85,NAVY2,RGBColor(0x23,0x40,0x5f),MSO_SHAPE.ROUNDED_RECTANGLE,1)
    txt(s,x+0.28,2.8,wc-0.5,0.4,[[(k,13,GOLDL,True)]])
    txt(s,x+0.28,3.2,wc-0.5,0.6,[[(big,21,WHITE,True)]])
    txt(s,x+0.28,3.85,wc-0.5,0.5,[[(sub,13,LT,False)]])
    x+=wc+0.14
txt(s,0.66,4.9,12,1.2,[[("The cost is measured. The models are built and out-of-sample. The logic is explainable end-to-end. ",19,LT,False),("The only decision left is to switch it on.",19,WHITE,True)]])
rect(s,0.66,6.7,12.0,0.012,NAVY2)
txt(s,0.66,6.78,7,0.3,[[("TAF · Toxic Account Forecasting",13,LT,False)]])
txt(s,7.0,6.78,5.66,0.3,[[("Zeal Group · Confidential · September 2026",13,LT,False)]],align=PP_ALIGN.RIGHT)

out=r"c:\Users\RoyVivasi\Documents\notebook\proposal\AntiFraud Solution - AI Driven Intelligence.pptx"
prs.save(out)
import os; print("PPTX:",out,os.path.getsize(out),"bytes,",len(prs.slides.__iter__.__self__._sldIdLst),"slides")
