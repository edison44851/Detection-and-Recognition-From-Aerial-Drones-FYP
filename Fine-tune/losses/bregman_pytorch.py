import torch

M_EPS = 1e-16


def sinkhorn(a, b, C, reg=1e-1, method='sinkhorn', maxIter=1000, tau=1e3,
             stopThr=1e-9, verbose=False, log=True, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    r"""
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    method : str
        method used for the solver either 'sinkhorn', 'greenkhorn', 'sinkhorn_stabilized' or
        'sinkhorn_epsilon_scaling', see those function for specific parameters
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------

    """

    # Always try the requested solver first. If numerical issues are
    # detected (NaN/Inf in the transport plan) and the requested solver is
    # not the stabilized variant, automatically fallback to the
    # `sinkhorn_stabilized` implementation and return diagnostics in the
    # log (when enabled).

    chosen = method.lower()

    # helper to call solvers uniformly and return (P, log_dict)
    def _call_solver(name):
        if name == 'sinkhorn':
            res = sinkhorn_knopp(a, b, C, reg, maxIter=maxIter,
                                 stopThr=stopThr, verbose=verbose, log=True,
                                 warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                                 **kwargs)
        elif name == 'sinkhorn_stabilized':
            res = sinkhorn_stabilized(a, b, C, reg, maxIter=maxIter, tau=tau,
                                      stopThr=stopThr, verbose=verbose, log=True,
                                      warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                                      **kwargs)
        elif name == 'sinkhorn_epsilon_scaling':
            res = sinkhorn_epsilon_scaling(a, b, C, reg,
                                           maxIter=maxIter, maxInnerIter=100, tau=tau,
                                           scaling_base=0.75, scaling_coef=None, stopThr=stopThr,
                                           verbose=False, log=True, warm_start=warm_start, eval_freq=eval_freq,
                                           print_freq=print_freq, **kwargs)
        else:
            raise ValueError("Unknown method '%s'." % method)

        # normalize output to (P, log_dict)
        _log: dict = {}
        if isinstance(res, tuple) and len(res) == 2:
            P, _maybe_log = res
            if isinstance(_maybe_log, dict):
                _log = _maybe_log
        else:
            P = res  # type: ignore[assignment]

        _log.setdefault('solver', name)
        # count NaN/Inf in the plan
        try:
            _P: torch.Tensor = P if isinstance(P, torch.Tensor) else P[0]  # type: ignore[index]
            nan_count = int(torch.isnan(_P).sum().item())
            inf_count = int(torch.isinf(_P).sum().item())
        except Exception:
            nan_count = -1
            inf_count = -1
        _log['nan_count'] = nan_count
        _log['inf_count'] = inf_count

        return P, _log

    P, _log = _call_solver(chosen)

    # If numerical issues found and not already stabilized, fallback
    if (_log.get('nan_count', 0) > 0 or _log.get('inf_count', 0) > 0) and chosen != 'sinkhorn_stabilized':
        if verbose:
            print(f"Numerical issues detected in solver '{chosen}' (NaN={_log['nan_count']}, Inf={_log['inf_count']}). Falling back to 'sinkhorn_stabilized'.")
        # call stabilized solver and merge diagnostics
        P_stab, stab_log = _call_solver('sinkhorn_stabilized')
        # attach fallback info
        stab_log['fallback_from'] = chosen
        # prefer stabilized plan
        P, _log = P_stab, stab_log

    # If caller did not request logs, only return P
    if not log:
        return P
    else:
        return P, _log


def sinkhorn_knopp(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9,
                   verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    r"""
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------

    """

    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    log_dict: dict = {'err': []} if log else {}

    if warm_start is not None:
        u = warm_start['u']
        v = warm_start['v']
    else:
        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    # compute K = exp(C / -reg)
    K = torch.div(C, -reg)
    K = torch.exp(K).to(device)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1

    # placeholders for intermediate results (avoid out= calls that may resize)
    KTu = None
    Kv = None

    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        # matrix-vector products (avoid out= to prevent resizing warnings)
        KTu = torch.matmul(u, K)
        v = torch.div(b, KTu + M_EPS)
        Kv = torch.matmul(K, v)
        u = torch.div(a, Kv + M_EPS)

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print('Warning: numerical errors at iteration', it)
            u, v = upre, vpre
            break

        if log and it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            # below is equivalent to:
            # b_hat = torch.sum(u.reshape(-1, 1) * K * v.reshape(1, -1), 0)
            # but with more memory efficient
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            # err = (b - b_hat).abs().sum().item()
            log_dict['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it = it + 1

    if log:
        log_dict['u'] = u
        log_dict['v'] = v
        log_dict['alpha'] = reg * torch.log(u + M_EPS)
        log_dict['beta'] = reg * torch.log(v + M_EPS)

    # transport plan
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    if log:
        return P, log_dict
    else:
        return P


def sinkhorn_stabilized(a, b, C, reg=1e-1, maxIter=1000, tau=1e3, stopThr=1e-9,
                        verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    r"""
    Solve the entropic regularization OT problem with log stabilization
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1]
    but with the log stabilization proposed in [3] an defined in [2] (Algo 3.1)

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    tau : float
        thershold for max value in u or v for log scaling
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    [2] Bernhard Schmitzer. Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. SIAM Journal on Scientific Computing, 2019
    [3] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    See Also
    --------

    """

    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    log_dict: dict = {'err': []} if log else {}

    if warm_start is not None:
        alpha = warm_start['alpha']
        beta = warm_start['beta']
    else:
        alpha = torch.zeros(na, dtype=a.dtype).to(device)
        beta = torch.zeros(nb, dtype=b.dtype).to(device)

    u = torch.ones(na, dtype=a.dtype).to(device) / na
    v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    def update_K(alpha, beta):
        """log space computation"""
        """memory efficient"""
        # K = exp((alpha + beta - C) / reg)
        K.copy_(torch.add(alpha.reshape(-1, 1), beta.reshape(1, -1)))
        K.add_(-C)
        K.div_(reg)
        K.copy_(torch.exp(K))

    def update_P(alpha, beta, u, v, ab_updated=False):
        """log space P (gamma) computation"""
        # P = exp((alpha + beta - C) / reg + log u + log v)
        P.copy_(torch.add(alpha.reshape(-1, 1), beta.reshape(1, -1)))
        P.add_(-C)
        P.div_(reg)
        if not ab_updated:
            P.add_(torch.log(u + M_EPS).reshape(-1, 1))
            P.add_(torch.log(v + M_EPS).reshape(1, -1))
        P.copy_(torch.exp(P))

    # preallocate K buffer and initialize
    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    update_K(alpha, beta)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1
    ab_updated = False

    # placeholders for intermediates (we'll assign results directly)
    KTu = None
    Kv = None
    P = torch.empty(C.shape, dtype=C.dtype).to(device)

    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        KTu = torch.matmul(u, K)
        v = torch.div(b, KTu + M_EPS)
        Kv = torch.matmul(K, v)
        u = torch.div(a, Kv + M_EPS)

        ab_updated = False
        # remove numerical problems and store them in K
        if u.abs().sum() > tau or v.abs().sum() > tau:
            alpha = alpha + reg * torch.log(u + M_EPS)
            beta = beta + reg * torch.log(v + M_EPS)
            u.fill_(1. / na)
            v.fill_(1. / nb)
            update_K(alpha, beta)
            ab_updated = True

        if log and it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            update_P(alpha, beta, u, v, ab_updated)
            b_hat = torch.sum(P, 0)
            err = (b - b_hat).pow(2).sum().item()
            log_dict['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it = it + 1

    if log:
        log_dict['u'] = u
        log_dict['v'] = v
        log_dict['alpha'] = alpha + reg * torch.log(u + M_EPS)
        log_dict['beta'] = beta + reg * torch.log(v + M_EPS)

    # transport plan
    update_P(alpha, beta, u, v, False)

    if log:
        return P, log_dict
    else:
        return P


def sinkhorn_epsilon_scaling(a, b, C, reg=1e-1, maxIter=100, maxInnerIter=100, tau=1e3, scaling_base=0.75,
                             scaling_coef=None, stopThr=1e-9, verbose=False, log=False, warm_start=None, eval_freq=10,
                             print_freq=200, **kwargs):
    r"""
    Solve the entropic regularization OT problem with log stabilization
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in [1] but with the log stabilization
    proposed in [3] and the log scaling proposed in [2] algorithm 3.2

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    tau : float
        thershold for max value in u or v for log scaling
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    [2] Bernhard Schmitzer. Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. SIAM Journal on Scientific Computing, 2019
    [3] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    See Also
    --------

    """

    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    if scaling_coef is None:
        scaling_coef = float(C.max()) + reg

    # scaling_coef is now guaranteed float; capture it so the closure is well-typed
    _scaling_coef: float = float(scaling_coef)

    def get_reg(it: int, reg: float, pre_reg: float) -> float:
        if it == 1:
            return _scaling_coef
        else:
            if (pre_reg - reg) * scaling_base < M_EPS:
                return reg
            else:
                return (pre_reg - reg) * scaling_base + reg

    it = 1
    err = 1.0
    running_reg: float = float(scaling_coef)

    log_dict: dict = {'err': []} if log else {}

    warm_start = None
    _log: dict = {}
    P: torch.Tensor = torch.zeros(1)  # initialised; overwritten on first loop iteration

    while (err > stopThr and it <= maxIter):
        running_reg = float(get_reg(it, reg, running_reg))
        _res = sinkhorn_stabilized(a, b, C, running_reg, maxIter=maxInnerIter, tau=tau,
                                   stopThr=stopThr, verbose=False, log=True,
                                   warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                                   **kwargs)
        P, _log = _res  # type: ignore[misc]

        warm_start = {}
        warm_start['alpha'] = _log['alpha']
        warm_start['beta'] = _log['beta']

        primal_val = (C * P).sum() + reg * (P * torch.log(P)).sum() - reg * P.sum()
        dual_val = (_log['alpha'] * a).sum() + (_log['beta'] * b).sum() - reg * P.sum()
        err = (primal_val - dual_val).item()
        if log:
            log_dict['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it = it + 1

    if log:
        log_dict['alpha'] = _log['alpha']
        log_dict['beta'] = _log['beta']
        return P, log_dict
    else:
        return P
