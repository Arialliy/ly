"""Logged pre-update telemetry only; not detection performance curves."""
from plot_common import *


def main():
    fig, axes=plt.subplots(2,2,figsize=(10.6,6.4))
    sources=[]; all_rows={}; runs={}
    specs=[("tanh_alpha",1.,"Signed residual scale, tanh($\\alpha$)"),
           ("applied_correction_to_input_norm",100.,"Applied correction / input norm (%)"),
           ("gate_saturation_fraction",100.,"Saturated spatial gates (%)")]
    for c,color,style,marker in [("S",COLORS[0],"-","o"),("T",COLORS[1],"--","s")]:
        p=RESULTS/f"training/{c}/run/TRAIN.jsonl"; rp=p.with_name("RUN.json")
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        run=json.loads(rp.read_text()); runs[c]=run
        assert len(rows)==run["final_step"]==run["steps"]
        assert all(row["finite"] for row in rows)
        telemetry=[r for r in rows if r.get("feature_statistics") and r.get("module_state_before_update")]
        all_rows[c]=telemetry; sources.extend([p,rp])
        x=[r["step"]-1 for r in telemetry]
        for ax,(key,scale,ylabel) in zip(axes.flat,specs):
            y=[r["feature_statistics"][key]*scale for r in telemetry]
            ax.plot(x,y,label=f"{c}-LFP",color=color,linestyle=style,marker=marker,markersize=3,linewidth=1.25)
            ax.set_ylabel(ylabel)
    assert runs["S"]["seed"]==runs["T"]["seed"]
    for ax,(key,scale,_) in zip(axes.flat,specs):
        if key!="tanh_alpha":
            ymax=max(r["feature_statistics"][key]*scale for rows in all_rows.values() for r in rows)
            ax.set_ylim(0,ymax*1.12 if ymax else 1)
    for c,key,label,color,style,marker in [
        ("S","center_kernel_norm","S: current",COLORS[4],"-","x"),
        ("T","previous_kernel_norm","T: previous",COLORS[2],"-","^"),
        ("T","center_kernel_norm","T: current",COLORS[0],"--","o"),
        ("T","next_kernel_norm","T: next",COLORS[1],":","s"),
    ]:
        rows=all_rows[c]
        axes[1,1].plot([r["step"]-1 for r in rows],
                       [r["module_state_before_update"][key] for r in rows],
                       label=label,color=color,linestyle=style,marker=marker,markersize=3,linewidth=1.25)
    axes[1,1].set_ylabel("Low-frequency gate kernel $L_2$ norm")
    axes[1,1].set_ylim(bottom=0)
    for i,ax in enumerate(axes.flat):
        ax.grid(color="#dddddd",lw=.45,zorder=0)
        ax.set_xlim(0,runs["S"]["steps"])
        ax.set_xlabel("Completed updates (pre-update telemetry)")
        ax.legend(frameon=False,loc="upper left" if i in [0,1,3] else "upper right")
        ax.text(-.025,1.025,f"({'abcd'[i]})",transform=ax.transAxes,weight="bold",fontsize=11)
    fig.subplots_adjust(left=.105,right=.99,bottom=.095,top=.95,hspace=.36,wspace=.29)
    finish(fig,"training_diagnostics",sources,{
        "seed":runs["S"]["seed"],"optimizer_updates_per_run":runs["S"]["steps"],
        "telemetry_samples_per_run":{c:len(v) for c,v in all_rows.items()},
        "x_definition":"logged step minus one; forward and module states are BEFORE that optimizer update",
        "x_max_in_telemetry":max(r["step"]-1 for rows in all_rows.values() for r in rows),
        "final512_checkpoint_state_is_not_plotted_as_telemetry":True,
        "line_segments_connect_recorded_samples_no_smoothing_or_interpolation_of_unrecorded_states":True,
        "gate_saturation_definition":"fraction with spatial sigmoid gate <0.01 or >0.99",
        "not_detection_performance_or_causal_evidence":True,
    })


if __name__ == "__main__":
    main()
