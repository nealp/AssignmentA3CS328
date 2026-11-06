"""Build the rubric report from saved, reproducible experiment evidence.

Requires reportlab. Run from this directory after report_evidence.py.
"""
import json
from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/report_evidence'
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=9.2,leading=13,spaceAfter=7,textColor=colors.HexColor('#243447')))
styles.add(ParagraphStyle(name='CellCustom',fontName='Helvetica',fontSize=7.4,leading=9.8))
styles.add(ParagraphStyle(name='CaptionCustom',fontSize=8,leading=11,textColor=colors.HexColor('#536576'),spaceAfter=8))
story=[]; md=[]

def para(text,style='BodyCustom'):
    story.append(Paragraph(escape(text),styles[style])); md.append(text+'\n')

def heading(text):
    story.append(Paragraph(escape(text),styles['Heading1'])); md.append('## '+text+'\n')

def sub(text):
    story.append(Paragraph(escape(text),styles['Heading2'])); md.append('### '+text+'\n')

def table(headers,rows,widths=None,padding=5):
    content=[[Paragraph(escape(str(x)),styles['CellCustom']) for x in r] for r in [headers]+rows]
    tab=Table(content,colWidths=widths or [504/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebed')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('BOTTOMPADDING',(0,0),(-1,-1),padding),('TOPPADDING',(0,0),(-1,-1),padding),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#238b8e')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f6f8')])]))
    story.extend([tab,Spacer(1,9)])
    md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |')
    md.extend('| '+' | '.join(map(str,r))+' |' for r in rows);md.append('')

def figure(path,caption,maxheight=260):
    path=ROOT/path; w,h=ImageReader(str(path)).getSize(); scale=min(504/w,maxheight/h)
    story.append(Image(str(path),width=w*scale,height=h*scale));para(caption,'CaptionCustom')
    md.append(f'![{caption}]({path.relative_to(ROOT)})\n')

def page(): story.append(PageBreak())
def pct(v): return f'{100*v:.2f}%'
def load(path): return json.loads((ROOT/path).read_text())

def main():
    counts=pd.read_csv(OUT/'count_comparison_test.csv')
    features=load('results/step4_features/summary.json')
    selected=load('results/step5_classification/frozen_selection.json')
    rep=load('results/step3_rep_counting/summary.json')
    selected_counter=rep['selected_counter']
    heading('Exercise recognition and rep counting')
    para('CS 328 | Assignment 3 | Exploration report')
    para('The final pipeline correctly classifies 82/129 test recordings (63.57%). Its rep-count MAE is 4.22, with 48.84% of recordings within two reported reps. The three design axes below were explored separately; better segmentation helps on average but can also remove the actual exercise set.')
    para('Evaluation protocol. The corpus has 430 recordings across eight exercises. The prescribed stratified recording split is 301 training / 129 test (random_state=42); training is split into 225 fit / 76 validation recordings. Counter settings were selected on fit data; features and models were selected using validation. Final classifiers were retrained on all 301 training recordings. No recording crosses splits. All test comparisons here are descriptive evaluations of frozen choices, not new tuning. MAE and within-two rates weight recordings equally; window accuracy weights windows equally. Rep labels are self-reported and may contain errors.')
    heading('1. Segmentation exploration')
    rows=[]
    descriptions={
        'none':('Whole recording','No trimming; diagnostic control'),
        'baseline':('Fixed threshold','1 s moving std > 0.5 m/s²; first to last active sample'),
        'adaptive':('Adaptive bouts','1 s std; q10 + 0.20(q90-q10), floor 0.02; gaps <=2 s; bouts >=4 s; pad 0.75 s'),
        'protocol_trim':('Protocol trim','Remove 5 s at both ends; retain full trace if duration <=11 s')}
    for name,(label,setting) in descriptions.items():
        r=counts[(counts.segmentation==name)&(counts.counter==(selected_counter if name=='adaptive' else 'selected'))].iloc[0]
        rows.append([label,setting,f'{r.MAE:.2f}',pct(r.within_2)])
    table(['Approach','Key parameters','Test MAE','Within +/-2'],rows,[91,265,65,83])
    para('Adaptive segmentation selects the longest surviving bout and falls back to the whole recording when none survives. It has the lowest test MAE (4.22 versus 4.71 for fixed threshold). Protocol trimming is essentially tied on MAE (4.25) and has a higher within-two rate (53.49%). This does not justify switching the development-selected method after seeing test results. Fixed threshold bridges distant handling bursts; protocol trimming assumes users followed the rest instructions; adaptive segmentation can miss low-amplitude reps or select sustained handling.')
    para('Classification effect (validation, fixed 100-tree forest, same 16 features and 6 s windows): adaptive 56.26%, baseline 53.35%, no segmentation 53.43%, protocol trim 54.91%. Adaptive improves on baseline by 2.91 percentage points, but the validation set is small and each segmentation changes the windows evaluated; these are pipeline comparisons, not paired identical-window tests.')
    figure(Path('results/step2_segmentation/figures/baseline_failure.png'),
        'Figure 1. Fit recording squats/002_2026-09-13_19-25-27.csv (Trouser, 10 reported reps). Baseline retains 0.63-36.30 s, including initial handling. Adaptive begins at 6.85 s, removing that burst, but still retains the large terminal burst.',175)
    page()
    heading('2. Classification exploration')
    sub('Window length and feature design')
    table(['Window length','Validation windows','Baseline-feature accuracy'],[[f"{r['window_sec']} s",r['validation_windows'],pct(r['validation_window_accuracy'])] for r in features['window_comparison']],[140,150,214])
    para('All windows use 50% overlap; segments shorter than a full window produce one shorter window. Six seconds was selected before model tuning because it gave the highest baseline-feature validation accuracy and usually captures several motion cycles. The 2.47-point difference between 2 s and 6 s is modest, rather than strong evidence of a universal optimum.')
    para('The seven baseline features describe filtered magnitude: mean, maximum, median, minimum, 25th percentile, 75th percentile, and standard deviation. Ten additions were tried; nine were kept. The table gives full 17-feature accuracy minus accuracy after removing that feature, in percentage points, using the same fixed forest. Effects depend on the other inputs and need not add together.')
    effects=pd.read_csv(ROOT/'results/step4_features/feature_effects.csv')
    explanation={
      'std_x':('Lat raises +4.00 recall points','Motion along x; sensitive to phone orientation'),
      'std_y':('Push-ups +12.24; lunges +4.76','Axis-specific motion amplitude'),
      'std_z':('Lat raises +24.00; lunges -7.94','Vertical/third-axis motion distribution'),
      'axis_dominance':('Lat raises +8.00; squats -4.68','Largest axis std / sum of axis stds'),
      'dominant_frequency':('Lat raises +4.00; squats -5.85','Largest-motion-axis FFT peak in 0.2-3 Hz'),
      'spectral_concentration':('Lat raises +12.00; push-ups +8.16','Peak-bin power / power in 0.2-3 Hz'),
      'corr_xy':('Lunges +12.70; lat raises +12.00','Coordination of x and y motion'),
      'corr_xz':('Lunges -11.11; lat raises +12.00','Tried and dropped; net validation decrease'),
      'corr_yz':('Push-ups +12.24; sit-ups -8.11','Coordination of y and z motion'),
      'rms_acceleration':('Lat raises +16.00; bicep curls +5.26','Root mean squared vector magnitude')}
    table(['Feature','Accuracy effect (pp)','Observed class recall effects','Rationale'],[[r.feature,f'{r.full_minus_without_accuracy_points:+.2f}',*explanation[r.feature]] for r in effects.itertuples()],[102,65,170,167])
    para('The baseline alone reaches 50.09%; axis motion 52.09%; rhythm alone 50.09%; correlations alone 53.54%; RMS alone 49.00%; axis + rhythm 55.35%; all ten additions 54.45%. Removing corr_xz gives the selected 16-feature set at 56.26%. Rhythm and axis features can help jointly even where individual removal effects are negative. Jumping jacks are already separable in the baseline, so the additions do not raise their validation recall in these ablations.')
    para('Fit-window medians support a physical interpretation: jumping jacks have RMS 26.38 m/s² and dominant frequency 1.33 Hz, versus squats at 2.36 m/s² and 0.67 Hz. Lat raises have corr_yz -0.38 versus squats +0.08. These are descriptive contrasts; placement variation and broad class overlap prevent universal thresholds.')
    page()
    sub('Models on the same frozen inputs')
    para('Both classifier types use adaptive segmentation, 6 s windows, and exactly the selected 16 features. Tree search: depths 3, 5, 8, 12, unlimited; gini/entropy; class_weight None/balanced. Forest search: 100/200 trees; depths unlimited/8/16; min_samples_leaf 1/3; class_weight None/balanced. All use random_state=42. Selection maximizes validation window accuracy, resolving exact ties by declared candidate order.')
    table(['Model','Key parameters','Validation window','Test window','Test recording'],[
      ['Decision Tree','depth 12; gini; no class weights','50.09%','46.49%','48.84% (63/129)'],
      ['Random Forest','200 trees; unlimited depth; leaf min 1; balanced','56.62%','60.43%','63.57% (82/129)']],[88,176,80,75,85])
    para('The forest was selected on validation, before test evaluation. Its test-window gain over the selected tree is 13.95 percentage points and recording gain is 14.73 points. Combining trees plausibly reduces dependence on a single set of thresholds, although the experiments do not isolate that mechanism. The default depth-5 entropy tree scores 48.43% on test windows, above the validation-selected tree; that observation does not reverse the selection. The default 100-tree forest scores 59.57%.')
    by=pd.read_csv(ROOT/'results/step5_classification/selected_models_test_by_exercise.csv')
    recalls=by.pivot(index='exercise',columns='model',values='recall')
    recording=load('results/step6_recordings/random_forest_recording_report_test.json')
    table(['Exercise','Tree window recall','Forest window recall','Change (pp)','Forest recording recall'],[[name,pct(r.decision_tree),pct(r.random_forest),f'{100*(r.random_forest-r.decision_tree):+.2f}',pct(recording[name]['recall'])] for name,r in recalls.iterrows()],[110,96,101,82,115])
    para('Gains occur for every exercise at window level but vary substantially: lat raises +25.64 points and push-ups +21.05, versus bicep curls +7.26. Ab crunches remain weak despite improving by 20.51 points. Class imbalance and scarce minority examples remain limitations; balancing was evaluated, not assumed to solve the problem.')
    para('One unweighted vote per window gives one prediction per recording; exact ties resolve alphabetically. The forest improves from 60.43% window accuracy to 63.57% recording accuracy, but ordinary window accuracy gives longer recordings more influence. Mean within-recording window accuracy is 59.04%, so voting also improves the equal-recording-weight reference by 4.53 points. Eleven forest recordings have tied votes; only four ties resolve correctly. Voting can remove isolated errors but cannot repair consistently wrong windows.')
    page()
    heading('3. Rep counting exploration')
    para('Every row below uses the final adaptive segmentation on all 129 test recordings. Parameters are the previously tried development candidates. Distance is a fixed number of seconds, not a fraction of an estimated period; prominence is a multiple of the filtered signal standard deviation. The baseline uses the prepared 5 Hz filtered magnitude; alternatives use second-order, zero-phase low-pass filtering before peak finding.')
    rows=[]
    for name,config in rep['candidate_settings'].items():
        r=counts[(counts.segmentation=='adaptive')&(counts.counter==name)].iloc[0]
        if config is None: signal='Magnitude'; params='5 Hz; height > mean; distance >=1 s'
        else:
            signal={'pca':'First PCA direction','axis':'Largest-std axis','magnitude':'Magnitude'}[config['signal_kind']]
            params=f"{config['cutoff_hz']:g} Hz; d={config['min_distance_sec']:g} s; p={config['prominence_fraction']:g} std"
        rows.append([('Selected' if name==selected_counter else name),signal,params,f'{r.MAE:.2f}',pct(r.within_2)])
    table(['Approach','Signal','Parameters','Test MAE','Within +/-2'],rows,[132,96,143,57,76])
    para('The selected PCA counter lowers MAE from 5.86 to 4.22 under the same adaptive segmentation, and raises within-two accuracy from 31.01% to 48.84%. Moving from 1.5 Hz to 1 Hz helps the axis/PCA variants by suppressing faster subcycles; increasing PCA prominence from 0.5 to 1 std at 0.6 s spacing lowers MAE from 4.87 to 4.22. Signal direction, filtering, and peak constraints work together; PCA alone is not the whole gain. The selected counter has a -1.16 rep bias, while the adaptive baseline overcounts by +1.15.')
    ex=pd.read_csv(OUT/'count_test_by_exercise.csv');pl=pd.read_csv(OUT/'count_test_by_placement.csv')
    table(['Exercise','n','MAE','Within +/-2'],[[r.exercise,r.recordings,f'{r.MAE:.2f}',pct(r.within_2)] for r in ex.itertuples()],[230,65,95,114])
    para('Lat raises (MAE 6.00) and ab crunches (5.75) have the largest class MAEs; push-ups have the lowest within-two rate (23.53%). Weakly coupled placements reduce observable motion, while multiple acceleration cycles per rep can still cause overcounting. By placement: Hand n=59, MAE 3.49, within-two 55.93%; Trouser n=60, 4.70, 43.33%; Jacket n=6, 6.50, 33.33%; Armband n=4, 4.25, 50.00%. No bag/backpack recordings occur in test. Small placement groups and different exercise mixes limit comparisons.')
    page()
    heading('4. Final pipeline and error attribution')
    para('Preprocessing: supplied preparation computes demeaned acceleration magnitude and applies a second-order 5 Hz zero-phase low-pass filter. Segmentation: adaptive 1 s moving magnitude std with threshold max(0.02, q10 + 0.20(q90-q10)); fill interior gaps <=2 s; discard bouts <4 s; retain longest bout with 0.75 s padding, or whole-recording fallback. The supplied cut helper also falls back if the cut is shorter than 1 s.')
    para('Classification: 6 s windows, 50% overlap; the seven baseline features plus std_x, std_y, std_z, axis_dominance, dominant_frequency, spectral_concentration, corr_xy, corr_yz, rms_acceleration. Dominant frequency uses a Hann-tapered FFT of the largest-std axis in 0.2-3 Hz. Model: RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1, class_weight="balanced", random_state=42), with other constructor defaults. Predict windows, then take the most frequent class with alphabetical tie resolution.')
    para('Rep counting: center x/y/z within the selected segment; second-order zero-phase 1 Hz low-pass each axis; project onto the first PCA direction with deterministic sign; count positive peaks at least 0.6 s apart and prominence >=1 signal std. Return zero if filtered signal std <0.01 m/s² or fewer than ten samples. These settings are common to every exercise; the exercise argument is unused, so counting does not depend on the true or predicted exercise label or phone-placement label.')
    figure(Path('results/step6_recordings/figures/random_forest_recording_confusion.png'),'Figure 2. Final recording-level test confusion matrix. Rows are true labels; columns are predictions. Total 129; diagonal total 82. Recording macro F1 = 0.591.',330)
    para('The following page accounts for every nonzero off-diagonal cell, including singletons. Attribution is diagnostic rather than a proven causal decomposition: there are no ground-truth set boundaries or interventions that uniquely identify the cause. Kept fraction is selected segment duration / recording duration; a very short fraction flags a possible segmentation failure, not proof. Feature-side explanations reflect motion/feature overlap; model-side evidence is strongest for documented voting ties.')
    page()
    sub('All nonzero recording-level confusions')
    errors=pd.read_csv(ROOT/'results/step6_recordings/final_classification_errors.csv')
    diag=pd.read_csv(OUT/'signal_diagnostics_test.csv')
    merged=errors.merge(diag,on='recording');merged['kept']=(merged.segment_end-merged.segment_start)/merged.duration
    groups=merged.groupby(['true','predicted']).agg(n=('recording','size'),kept=('kept','median'))
    mechanisms={
      ('Ab crunches','Push-ups'):('Segmentation','Very short selected bouts may retain handling instead of weak torso motion.'),
      ('Ab crunches','Squats'):('Feature side','Gravity-free torso/pocket motion lacks a reliable posture cue.'),
      ('Bicep curls','Ab crunches'):('Feature side','Jacket motion is an indirect measure of arm flexion.'),
      ('Bicep curls','Lat raises'):('Feature side','Both are arm movements with overlapping amplitude summaries.'),
      ('Bicep curls','Lunges'):('Segmentation','Only 21% of this trouser recording is retained; much of the arm set may be absent.'),
      ('Bicep curls','Push-ups'):('Feature side','Magnitude and direction summaries can overlap across placements.'),
      ('Bicep curls','Squats'):('Feature side','Mixed phone orientation and cadence allow overlap with squat windows.'),
      ('Lat raises','Jumping jacks'):('Feature side','Strong arm acceleration can resemble the arm phase of jumping jacks.'),
      ('Lat raises','Squats'):('Segmentation','Only 9% is retained from a trouser recording; weak arm motion can be missed.'),
      ('Lunges','Push-ups'):('Feature side','Cyclic motion summaries do not encode a unique body posture.'),
      ('Lunges','Squats'):('Feature side','Both involve repeated leg flexion and similar pocket motion.'),
      ('Push-ups','Ab crunches'):('Feature side','Trouser placement measures torso/pelvic motion in both exercises.'),
      ('Push-ups','Jumping jacks'):('Feature side','Weak armband coupling gives unreliable class rhythm and intensity cues.'),
      ('Push-ups','Lunges'):('Feature side','Trouser motion overlaps in amplitude and temporal summaries.'),
      ('Push-ups','Sit-ups'):('Segmentation','Short selected bouts can discard the lower-amplitude exercise cycles.'),
      ('Sit-ups','Ab crunches'):('Feature side','Similar torso flexion lacks posture information; one error also comes from a 1-1 vote tie.'),
      ('Sit-ups','Jumping jacks'):('Feature side','The full trace is kept despite low global axis motion; summaries can be misleading.'),
      ('Sit-ups','Lunges'):('Feature side','Repeated hand/torso movement can share the same amplitude and cadence features.'),
      ('Sit-ups','Push-ups'):('Segmentation','Only 27% is retained, suggesting the selected bout misses much of the set.'),
      ('Sit-ups','Squats'):('Feature side','The mostly retained trace still lacks an explicit lying/standing posture cue.'),
      ('Squats','Ab crunches'):('Feature side','Relatively weak motion and gravity removal leave overlapping torso descriptors.'),
      ('Squats','Jumping jacks'):('Feature side','High-amplitude hand movement can resemble a more vigorous exercise.'),
      ('Squats','Lat raises'):('Feature side','Jacket movement reflects garment/phone dynamics as well as the exercise.'),
      ('Squats','Lunges'):('Feature side','Closely related leg cycles share cadence and motion amplitude.'),
      ('Squats','Push-ups'):('Feature side','Different placements can produce similar cyclic acceleration summaries.'),
      ('Squats','Sit-ups'):('Feature side / model side','Torso summaries overlap; one documented 3-3 window tie resolves to sit-ups.')}
    rows=[]
    for (true,pred),r in groups.iterrows():
        axis,why=mechanisms[(true,pred)]
        rows.append([true+' → '+pred,int(r.n),f'{r.kept*100:.0f}%',axis,why])
    table(['True to predicted','n','Median kept','Attribution','Likely mechanism (hypothesis)'],rows,[111,20,42,77,254],padding=3)
    para('Class scarcity may contribute to some model errors (especially six test lat-raise and eight ab-crunch recordings), but the matrix alone cannot demonstrate separable features followed by a faulty model boundary. Accordingly, most entries remain feature-side or segmentation hypotheses rather than confident model-side causal claims. All 47 errors are included above.')
    page()
    heading('5. Top failure modes')
    sub('Failure 1: a handling bout replaces the exercise set')
    para('Specific test recording: data/push_ups/004_2026-09-14_14-58-52.csv, Trouser, 20 reported reps. The final segment is 3.14-9.40 s and produces only 3 peaks (error -17), the largest absolute final-counter error on test. The final classifier also predicts sit-ups for this recording.')
    figure(Path('results/report_evidence/failure_counting.png'),'Figure 3. The selected segment covers the initial large burst. Repeated lower-amplitude motion continues after the selected end; the lower panel shows the actual three detected PCA peaks.',265)
    para('Physical mechanism. Large early motion raises the recording-relative threshold, while weak pocket motion during push-ups fails to form the chosen sustained bout. The visible repeated motion around 10-28 s is excluded; its exact rep boundaries cannot be established from self-reported counts alone. The counter only sees the retained handling signal, so the downstream undercount is primarily a segmentation problem.')
    para('Specific change to evaluate on future development data: score candidate bouts using sustained periodicity as well as energy, allowing a lower-energy repeated set to outrank phone handling. This is a proposed improvement, not an implemented or test-tuned fix.')
    sub('Why this is more than a peak-threshold problem')
    para('Lowering prominence within the selected six-second fragment cannot restore the omitted set. This example separates an upstream boundary failure from a rep-counting failure within a correctly selected set. Conversely, squats have a positive mean count bias (+1.97), consistent with residual multiple cycles per reported rep; not every count error has the same cause.')
    page()
    sub('Failure 2: squats and sit-ups overlap, then voting breaks a tie')
    para('Specific test recording: data/squats/024_2026-10-14_22-16-05.csv, Trouser, 10 reported reps. The selected interval is 5.10-26.65 s. Three of six windows predict squats and three predict sit-ups, so alphabetical tie resolution outputs sit-ups. The final counter finds 20 peaks, illustrating that classification and count errors can coexist.')
    figure(Path('results/report_evidence/failure_classification.png'),'Figure 4. The selected segment includes the sustained central motion, with 20 detected PCA peaks. A plausible set interval does not guarantee unique exercise features or one peak per reported rep.',235)
    ft=pd.read_csv(ROOT/'results/step5_classification/features_test.csv')
    fitmed=pd.read_csv(ROOT/'results/step4_features/feature_medians_fit_by_exercise.csv').set_index('exercise')
    example=ft[ft.recording=='data/squats/024_2026-10-14_22-16-05.csv'].median(numeric_only=True)
    table(['Feature','Example median','Fit squats median','Fit sit-ups median'],[[name,f'{example[name]:.3f}',f'{fitmed.loc["Squats",name]:.3f}',f'{fitmed.loc["Sit-ups",name]:.3f}'] for name in ['std','std_z','corr_xy','rms_acceleration']],[150,118,118,118])
    para('Physical mechanism. A trouser phone measures cyclic pelvic/torso acceleration; after gravity removal, these features lack an explicit lying-versus-standing posture cue. The example corr_xy (0.229) is nearly the sit-up fit median (0.230), while magnitude std (1.107) is near the squat median (1.082), so different features support different classes. Class medians do not prove distribution overlap, but support the mixed-cue interpretation.')
    para('Attribution: feature side is the likely general limitation behind squats-to-sit-ups confusion (four recordings); model-side aggregation is the directly observed cause of this example’s final tied decision. Segmentation appears plausible in the trace, but no annotated boundary is available. A specific improvement is to compare probability-weighted voting against hard voting on development recordings; it may resolve conflicting-window ties but cannot guarantee better performance.')
    heading('6. Contribution statement')
    para('Aditya Sathishkumar, Neal Patel, and Jayant Manem all contributed to the code and exploration report.')
    para('Evidence and reproducibility. Sections 1-3 use saved Step 2-5 development results and results/report_evidence test comparisons; Sections 4-5 use Step 6 recording predictions and the traces generated by report_evidence.py. Reproduce the new test counting tables with python3 report_evidence.py and rebuild this document with python3 build_exploration_report.py (reportlab required). Existing models, source methods, split files, and util.py were not modified to write this report.')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#c9d6dd'));canvas.line(54,41,558,41)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#536576'))
        canvas.drawString(54,28,'CS 328 | A3 Exploration');canvas.drawRightString(558,28,str(doc.page))
    doc=SimpleDocTemplate(str(ROOT/'exploration_report.pdf'),pagesize=(612,792),leftMargin=54,rightMargin=54,topMargin=40,bottomMargin=53,
                          title='Exercise recognition and rep counting - Exploration report',author='CS 328 project group')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'exploration_report.md').write_text('\n'.join(md))
    print('Created exploration_report.pdf and exploration_report.md')


if __name__ == '__main__': main()
