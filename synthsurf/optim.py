import torch



class ConjugateGradient:
    """
    Conjugate-gradient solver for linear systems of the form `A @ x = b`
    """

    def __init__(self, max_iter=32, tol=1e-4, dot=None,
                 target=None, forward=None, precond=None):
        """

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations
        tol : float
            Tolerance for early stopping
        dot : callable(tensor, tensor) -> tensor
            Function to use for dot product
        """
        self.max_iter = max_iter
        self.tol = tol
        self.dot = dot
        self.target = target
        self.forward = forward
        self.precond = precond

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, target=None, forward=None, precond=None, init=None, inplace=False):
        """Solve for `x` in `A @ x = b`

        Parameters
        ----------
        target : tensor
            Target vector `b`
        forward : callable(tensor) -> tensor
            Forward matrix-vector product `A`
        precond : callable(tensor) -> tensor, optional
            Preconditioning matrix-vector product `P`
        init : tensor, optional
            Initial value for the solution `x`

        Returns
        -------
        solution : tensor
            Solution `x`

        """
        x = (torch.zeros_like(target) if init is None
             else init if inplace else init.clone())
        g = target if target is not None else self.target
        A = forward or self.forward
        P = precond or self.precond or torch.clone
        dot = self.dot or _dot

        # init
        r = A(x)                    # r  = A @ x
        torch.sub(g, r, out=r)      # r  = g - r
        z = P(r)                    # z  = P @ r
        rz = dot(r, z)              # rz = r' @ z
        p = z.clone()               # Initial conjugate directions p

        for n_iter in range(self.max_iter):
            Ap = A(p)                       # Ap = A @ p
            pAp = dot(p, Ap)                # p' @ A @ p
            a = rz / pAp.clamp_min_(1e-12)  # α = (r' @ z) / (p' @ A @ p)

            if self.tol:
                obj = a * (a * pAp + 2 * dot(p, r))
                obj *= a.numel() / r.numel()
                obj = obj.mean()
                if obj < self.tol:
                    break

            x.addcmul_(a, p)                # x += α * p
            r.addcmul_(a, Ap, value=-1)     # r -= α * (A @ p)
            z = P(r)                        # z  = P @ r
            rz0 = rz
            rz = dot(r, z)                  # rz = r' @ z
            b = rz / rz0                    # β
            torch.addcmul(z, b, p, out=p)   # p = z + β * p

        return x

    def solve_(self, init, target=None, forward=None, precond=None):
        """Solve for `x` in `A @ x = b` in-place.

        Parameters
        ----------
        init : tensor
            Initial value for the solution `x`
        target : tensor
            Target vector `b`
        forward : callable(tensor) -> tensor
            Forward matrix-vector product `A`
        precond : callable(tensor) -> tensor, optional
            Preconditioning matrix-vector product `P`

        Returns
        -------
        solution : tensor
            Solution `x`

        """
        return self.solve(target, forward, precond, init, inplace=True)

    def advanced_solve_(self, init, target=None, forward=None, precond=None,
                        p=None, r=None, z=None):
        """Solve for `x` in `A @ x = b`

        Parameters
        ----------
        init : tensor
            Initial value for the solution `x`
        target : tensor
            Target vector `b`
        forward : callable(tensor, out=tensor) -> tensor
            Forward matrix-vector product `A`
        precond : callable(tensor, out=tensor) -> tensor, optional
            Preconditioning matrix-vector product `P`

        Returns
        -------
        solution : tensor
            Solution `x`

        """
        x = init
        g = target if target is not None else self.target
        A = forward or self.forward
        P = precond or self.precond or (lambda z, out: z)
        p = torch.empty_like(x) if p is None else p
        r = torch.empty_like(x) if r is None else r
        z = torch.empty_like(x) if z is None else z
        Ap = z  # can use same buffer as z
        dot = self.dot or _dot

        # init
        r = A(x, out=r)                 # r  = A @ x
        r = torch.sub(g, r, out=r)      # r  = g - r
        z = P(r, out=z)                 # z  = P @ r
        rz = dot(r, z)                  # rz = r' @ z
        p = p.copy_(z)                  # Initial conjugate directions p

        for n_iter in range(self.max_iter):
            Ap = forward(p, out=Ap)         # Ap = A @ p    (can use same buffer as z)
            pAp = dot(p, Ap)                # p' @ A @ p
            a = rz / pAp.clamp_min(1e-12)   # α = (r' @ z) / (p' @ A @ p)

            if True:  # self.tol:
                obj = a * (a * pAp + 2 * dot(p, r))
                obj *= a.numel() / r.numel()
                obj = obj.mean()
                print(obj)
                # if obj < self.tol:
                #     break

            x.addcmul_(a, p)                # x += α * p

            if n_iter == self.max_iter-1:
                break

            r.addcmul_(a, Ap, value=-1)     # r -= α * (A @ p)
            z = P(r, out=z)                 # z  = P @ r
            rz0, rz = rz, dot(r, z)         # rz = r' @ z
            b = rz / rz0                    # β
            torch.addcmul(z, b, p, out=p)   # p = z + β * p

        return x


def _dot(x, y, out=None):
    return torch.dot(x.flatten(), y.flatten(), out=out)