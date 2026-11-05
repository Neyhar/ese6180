"""Inspect exactly what a partial-observation learner sees and applies."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import exact_policy, history_feature
from evaluation import deploy_trace
from pitfalls import write_csv


def save_learner_view(system, gain, policies, out, steps, warmup):
    observer=exact_policy(system,gain,2,1)
    policies=policies+[('known-model observer',observer,1)]
    reconstruction=system.reconstruction(2)
    records,features=[],[]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for name,policy,ell in policies:
        pert=deploy_trace(system,gain,policy,ell,[.001,1.],steps,warmup)
        support=deploy_trace(system,gain,policy,ell,[0.,1.],steps,warmup)
        errors=[];action_errors=[]
        for j in range(steps):
            t=warmup+j
            feat2=history_feature(pert['y'],pert['u'],t,2)
            estimate=reconstruction@feat2
            x=pert['x'][t];u=pert['u'][t]
            errors.append(abs(estimate[0]-x[0]));action_errors.append(np.linalg.norm(u-gain@x))
            records.append(dict(controller=name,step=j,hidden_x1=float(x[0]),observed_y=float(pert['y'][t,0]),
                                model_estimated_x1=float(estimate[0]),actual_u1=float(u[0]),actual_u2=float(u[1]),
                                action_error_at_learner_state=float(action_errors[-1])))
            if j%ell==0:
                feat=history_feature(pert['y'],pert['u'],t,policy.context)
                if j==0:
                    other=history_feature(support['y'],support['u'],t,policy.context)
                    np.testing.assert_allclose(feat,other,atol=1e-14)
                features.append(dict(controller=name,step=j,context=policy.context,
                                     observation_history=json.dumps(pert['y'][t-policy.context+1:t+1].reshape(-1).tolist()),
                                     applied_action_history=json.dumps(pert['u'][t-policy.context+1:t].reshape(-1).tolist()),
                                     input_feature=json.dumps(feat.tolist())))
        xs=pert['x'][warmup:]
        ys=pert['y'][warmup:]-support['y'][warmup:]
        axes[0,0].semilogy(np.maximum(abs(xs[:,0]),1e-16),label=name)
        axes[0,1].semilogy(np.maximum(abs(ys[:,0]),1e-16),label=name)
        axes[1,0].semilogy(np.maximum(errors,1e-16),label=name)
        axes[1,1].semilogy(np.maximum(action_errors,1e-16),label=name)
    labels=[('Hidden state departure','|x₁|'),('Visible departure from paired support rollout','|Δy|, y=x₂'),
            ('Information available with a known model','|reconstructed x₁ − true x₁|'),('What the deployed policy does','||u − Kx||₂')]
    for ax,(title,label) in zip(axes.flat,labels):
        ax.set(title=title,xlabel='Deployment step',ylabel=label);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Partial observation: initial x₁ jump is invisible at step 0; its effect reaches y at step 1\nModel reconstruction is a diagnostic/baseline; learned ridge policies are not given the reconstructed state.',fontsize=12)
    fig.savefig(out/'learner_view.png',dpi=180);plt.close(fig)
    write_csv(records,out/'learner_view.csv');write_csv(features,out/'learner_inputs.csv')
