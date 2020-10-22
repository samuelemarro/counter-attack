import torch


def main():

    def profile_onthefly(func):
        def _wrapper(*args, **kw):
            import line_profiler
            from six.moves import cStringIO
            profile = line_profiler.LineProfiler()

            result = profile(func)(*args, **kw)

            file_ = cStringIO()
            profile.print_stats(stream=file_, stripzeros=True)
            file_.seek(0)
            text = file_.read()
            print(text)
            return result
        return _wrapper

    def test_syncpoint_mask(data, conv2d, mask, do_mask=False, do_sync=False):
        print('---------')
        print(' * do_mask = {!r}'.format(do_mask))
        print(' * do_sync = {!r}'.format(do_sync))

        torch.cuda.synchronize()

        x = data
        N = 10
        # Do some busy work
        for i in range(N):
            x = conv2d(x)

        # Create a mask
        x = x.view(-1)
        if do_mask:
            a = x.sum()#x = #torch.sign(torch.relu(x))#x[mask] = 0.0

        if do_sync:
            torch.cuda.synchronize()

        # Do more busy work
        for i in range(N):
            x = x * x
            x = torch.sqrt(x)

        torch.cuda.synchronize()

    xpu = torch.device('cuda:0')
    # Setup dummy data
    bsize = 30
    data = torch.rand(bsize, 3, 512, 512).to(xpu)
    conv2d = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1).to(xpu)

    
    mask = (data > .5).view(-1)
    mask = torch.nonzero(mask)
    print(mask.dtype)

    # Profile to show how masking causes an implicity sync-point

    test_syncpoint_mask(data, conv2d, mask, do_mask=False, do_sync=False)

    profile_onthefly(test_syncpoint_mask)(data, conv2d, mask, do_mask=True, do_sync=False)

    profile_onthefly(test_syncpoint_mask)(data, conv2d, mask, do_mask=False, do_sync=False)

    profile_onthefly(test_syncpoint_mask)(data, conv2d, mask, do_mask=False, do_sync=True)

def test_synpoint_other(opname, op):
    def test_syncpoint_(data, conv2d):
        print('---------')
        print('opname = {!r}'.format(opname))
        torch.cuda.synchronize()
        x = data
        N = 10
        for i in range(N):
            x = conv2d(x)  # non-syncing busy work
        op(x)
        # Do more busy work
        for i in range(N):
            x = torch.sqrt(x * x)  # non-syncing busy work
        torch.cuda.synchronize()
    profile_onthefly(test_syncpoint_)(data, conv2d)

main()