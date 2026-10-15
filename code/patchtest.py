import benchmark as B, math, numpy as np
A=B.A

# ---- FIX 1: min_obs_dist must start at +inf (11.py line 536) ----
def _evaluate(self, traj, local_goal, global_path, obstacles):
    te=traj[-1]
    dg=math.hypot(te[0]-local_goal[0],te[1]-local_goal[1])
    hg=math.atan2(local_goal[1]-self.uav.pos[1],local_goal[0]-self.uav.pos[0])
    th=math.atan2(te[1]-traj[0][1],te[0]-traj[0][0])
    hd=abs(hg-th); hd=min(hd,2*math.pi-hd)
    heading_score=1.0-hd/math.pi
    min_obs=float('inf')                      # was 0.0
    for p in traj:
        for (ox,oy,orr) in obstacles:
            min_obs=min(min_obs,math.hypot(p[0]-ox,p[1]-oy)-self.uav.radius-orr)
    if min_obs==float('inf'): min_obs=99.0
    safe=3.0
    obs_score=(1.0-min_obs/safe) if min_obs<safe else 0.0
    return 1.0*heading_score - 2.0*obs_score + 0.5/(dg+1.0)

# ---- FIX 2: DWA margin 0.3 -> 0.5, matching the paper's collision criterion ----
def _sim(self,v,omega,obstacles):
    if not obstacles:
        return [(self.uav.pos[0],self.uav.pos[1]),(self.uav.pos[0]+v*0.3,self.uav.pos[1]+v*0.3)]
    traj=[(self.uav.pos[0],self.uav.pos[1])]
    x,y=float(self.uav.pos[0]),float(self.uav.pos[1]); h=float(self.uav.heading)
    for _ in range(int(self.predict_time/self.dt)):
        x+=v*math.cos(h)*self.dt; y+=v*math.sin(h)*self.dt; h+=omega*self.dt
        traj.append((x,y))
        for (ox,oy,orr) in obstacles:
            if math.hypot(x-ox,y-oy) < self.uav.radius+orr+0.5:   # was +0.3
                return None
    return traj

A.DynamicWindowApproach._evaluate=_evaluate
A.DynamicWindowApproach._simulate_with_collision_check=_sim

# ---- FIX 3: validate the spline; the smoother must not undo the LOS check ----
def clear(pts,env,uav):
    for x,y in pts:
        for o in env.static_obstacles:
            if math.hypot(x-o.x,y-o.y) < uav.radius+o.radius+0.5: return False
    return True

def capped_remove(path,env,thr,max_seg):
    if len(path)<3: return path
    out=[path[0]]; i=0
    while i < len(path)-1:
        j=len(path)-1; adv=False
        while j>i+1:
            if math.hypot(path[j][0]-path[i][0],path[j][1]-path[i][1])<=max_seg \
               and A.PathSmoother.is_line_safe(env,path[i],path[j],thr):
                out.append(path[j]); i=j; adv=True; break
            j-=1
        if not adv:
            i+=1
            if i<len(path): out.append(path[i])
    return out

def safe_smooth(raw,env,uav,thr=0.7):
    for cap in (1e9,30.0,20.0,12.0,8.0):
        red=capped_remove(raw,env,thr,cap)
        sp=A.PathSmoother.cubic_spline_smooth(red,num_points=150)
        if clear(sp,env,uav): return sp,cap,len(red)
    return red,None,len(red)

def probe(seed,fixed=True):
    env,uav=B.build_scenario(seed); sm=A.PathSmoother()
    dwa=A.DynamicWindowApproach(uav,env,B.GOAL)
    env.compute_risk_grid(uav,0.0,use_kf=True)
    def replan(t):
        env.update_kf_trackers(t); env.compute_risk_grid(uav,t,use_kf=True)
        raw=A.AdaptiveAStar(env,uav,B.GOAL,risk_threshold=0.6).run()
        if fixed:
            sp,_,_=safe_smooth(raw,env,uav); return sp
        red=sm.remove_redundant(raw,env,risk_threshold=0.7)
        return sm.cubic_spline_smooth(red,num_points=150)
    path=replan(0.0); t=0.0;last=0.0;act=False;ast=0.0;hi=0;L=0.0;prev=tuple(uav.pos)
    for _ in range(2000):
        t+=0.2
        env.update_dynamic(0.2); env.update_kf_trackers(t)
        env.compute_risk_grid(uav,t,use_kf=True); env.compute_eval_risk_grid(uav)
        for o in list(env.static_obstacles)+list(env.dynamic_obstacles):
            if math.hypot(uav.pos[0]-o.x,uav.pos[1]-o.y)<uav.radius+o.radius+0.5:
                return ('DYN' if o.is_dynamic else 'STATIC'),t,L
        if uav.distance_to(B.GOAL)<5.0: return 'SUCCESS',t,L
        r=B.mean_risk_around(env,uav)
        hi=hi+1 if r>0.3 else 0
        if hi>=2 and not act: act=True;ast=t;hi=0
        if act and t-ast>3.0 and r<0.15: act=False
        tgt=B.find_lookahead(path,uav.pos,3.0 if act else 8.0,10 if act else 25)
        v,w=dwa.calculate_velocity(path,tgt); uav.update(v,w,0.2)
        cur=tuple(uav.pos); L+=math.hypot(cur[0]-prev[0],cur[1]-prev[1]); prev=cur
        if t-last>=5.0: path=replan(t); last=t
    return 'TIMEOUT',t,L

from collections import Counter
c=Counter()
for s in range(10):
    res,t,L=probe(s,fixed=True)
    c[res]+=1
    print('seed %2d: %-8s t=%5.1fs len=%6.1fm'%(s,res,t,L))
print(); print('FIXED:',dict(c))
