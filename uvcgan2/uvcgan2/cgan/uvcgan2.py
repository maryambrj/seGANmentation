# pylint: disable=not-callable
# NOTE: Mistaken lint:
# E1102: self.criterion_gan is not callable (not-callable)

import itertools
import torch

from torchvision.transforms import GaussianBlur, Resize

from uvcgan2.torch.select            import (
    select_optimizer, extract_name_kwargs
)
from uvcgan2.torch.queue             import FastQueue
from uvcgan2.torch.funcs             import prepare_model, update_average_model
from uvcgan2.torch.layers.batch_head import BatchHeadWrapper, get_batch_head
from uvcgan2.base.losses             import GANLoss, IoULoss, DiceCoefficient
from uvcgan2.torch.gradient_penalty  import GradientPenalty
from uvcgan2.models.discriminator    import construct_discriminator
from uvcgan2.models.generator        import construct_generator

from .model_base import ModelBase
from .named_dict import NamedDict
from .funcs import set_two_domain_input

def construct_consistency_model(consist, device):
    name, kwargs = extract_name_kwargs(consist)

    if name == 'blur':
        return GaussianBlur(**kwargs).to(device)

    if name == 'resize':
        return Resize(size = 32, **kwargs).to(device)  #Maryam

    raise ValueError(f'Unknown consistency type: {name}')

def queued_forward(batch_head_model, input_image, queue, update_queue = True):
    output, pred_body = batch_head_model.forward(
        input_image, extra_bodies = queue.query(), return_body = True
    )

    if update_queue:
        queue.push(pred_body)

    return output

class UVCGAN2(ModelBase):
    # pylint: disable=too-many-instance-attributes
    def _setup_images(self, _config):
        images = [
            'real_a', 'real_b',
            'fake_a', 'fake_b',
            'reco_a', 'reco_b',
            'consist_real_a', 'consist_real_b',
            'consist_fake_a', 'consist_fake_b',
        ]
        # images = [
        #     'real_a', 'real_b',
        #     'fake_b',
        #     'consist_real_a',
        #     'consist_fake_b',
        # ]

        if self.is_train and self.lambda_idt > 0:
            images += [ 'idt_a', 'idt_b', ]
            # images += [ 'idt_b', ]

        return NamedDict(*images)

    def _construct_batch_head_disc(self, model_config, input_shape):
        disc_body = construct_discriminator(
            model_config, input_shape, self.device
        )

        disc_head = get_batch_head(self.head_config)
        disc_head = prepare_model(disc_head, self.device)

        return BatchHeadWrapper(disc_body, disc_head)

    def _setup_models(self, config):
        models = {}

        shape_a = config.data.datasets[0].shape
        shape_b = config.data.datasets[1].shape

        models['gen_ab'] = construct_generator(
            config.generator, shape_a, shape_b, self.device
        )
        models['gen_ba'] = construct_generator(
            config.generator, shape_b, shape_a, self.device
        )

        if self.avg_momentum is not None:
            models['avg_gen_ab'] = construct_generator(
                config.generator, shape_a, shape_b, self.device
            )
            models['avg_gen_ba'] = construct_generator(
                config.generator, shape_b, shape_a, self.device
            )

            models['avg_gen_ab'].load_state_dict(models['gen_ab'].state_dict())
            models['avg_gen_ba'].load_state_dict(models['gen_ba'].state_dict())

        if self.is_train:
            models['disc_a'] = self._construct_batch_head_disc(
                config.discriminator, config.data.datasets[0].shape
            )
            models['disc_b'] = self._construct_batch_head_disc(
                config.discriminator, config.data.datasets[1].shape
            )

        return NamedDict(**models)

    def _setup_losses(self, config):
        losses = [
            'gen_ab', 'gen_ba', 'disc_a', 'disc_b'
        ]
        # losses = [
        #     'gen_ab', 'disc_b'
        # ]
        if self.is_train and self.lambda_a > 0:
            losses += [ 'cycle_a']

        if self.is_train and self.lambda_b > 0:
            losses += [ 'cycle_b']

        if self.is_train and self.lambda_iou > 0:
            losses += [ 'iou']

        if self.is_train and self.lambda_dice > 0:
            losses += [ 'dice']

        if self.is_train and self.lambda_idt > 0:
            losses += [ 'idt_a', 'idt_b' ]
            # losses += [ 'idt_b']

        if self.is_train and config.gradient_penalty is not None:
            losses += [ 'gp_a', 'gp_b' ]
            # losses += [ 'gp_b' ] # grad penalty loss is for discriminator

        if self.consist_model is not None:
            losses += [ 'consist_a', 'consist_b' ]
            # losses += [ 'consist_a' ]

        return NamedDict(*losses)

    def _setup_optimizers(self, config):
        optimizers = NamedDict('gen', 'disc')

        optimizers.gen = select_optimizer(
            itertools.chain(
                self.models.gen_ab.parameters(),
                self.models.gen_ba.parameters()
            ),
            config.generator.optimizer
        )

        optimizers.disc = select_optimizer(
            itertools.chain(
                self.models.disc_a.parameters(),
                self.models.disc_b.parameters()
            ),
            config.discriminator.optimizer
        )

        return optimizers

    def __init__(
        self, savedir, config, is_train, device, head_config = None,
        lambda_a        = 1.0,
        lambda_b        = 1.0,
        lambda_iou      = 1.0,
        lambda_dice     = 0,
        lambda_idt      = 0,
        lambda_consist  = 0,
        head_queue_size = 3,
        avg_momentum    = None,
        consistency     = None,
        # consistency     = 'resize',
    ):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals
        self.lambda_a       = lambda_a
        self.lambda_b       = lambda_b
        self.lambda_iou     = lambda_iou
        self.lambda_dice    = lambda_dice
        self.lambda_idt     = lambda_idt
        self.lambda_consist = lambda_consist
        self.avg_momentum   = avg_momentum
        self.head_config    = head_config or {}
        self.consist_model  = None
        # self.consist_model  = consistency

        if (lambda_consist > 0) and (consistency is not None):
            self.consist_model \
                = construct_consistency_model(consistency, device)

        assert len(config.data.datasets) == 2, \
            "CycleGAN expects a pair of datasets"

        super().__init__(savedir, config, is_train, device)

        self.criterion_gan     = GANLoss(config.loss).to(self.device)
        self.criterion_cycle_a = torch.nn.L1Loss()
        # self.criterion_cycle_a = torch.nn.MSELoss()  # or L2 loss
        self.criterion_cycle_b = IoULoss().to(self.device)
        # self.criterion_cycle_b = DiceCoefficient().to(self.device)
        self.iou_loss          = IoULoss().to(self.device)
        self.dice_loss         = DiceCoefficient().to(self.device)
        self.criterion_idt     = torch.nn.L1Loss()
        self.criterion_consist = torch.nn.L1Loss()

        if self.is_train:
            self.queues = NamedDict(**{
                name : FastQueue(head_queue_size, device = device)
                    # for name in [ 'real_a', 'real_b', 'fake_b' ]
                    for name in [ 'real_a', 'real_b', 'fake_a', 'fake_b' ]
            })

            self.gp = None

            if config.gradient_penalty is not None:
                self.gp = GradientPenalty(**config.gradient_penalty)

    def _set_input(self, inputs, domain):
        set_two_domain_input(self.images, inputs, domain, self.device)

        if self.images.real_a is not None:
            if self.consist_model is not None:
                self.images.consist_real_a \
                    = self.consist_model(self.images.real_a)
        
        if self.images.real_b is not None:
            if self.consist_model is not None:
                self.images.consist_real_b \
                    = self.consist_model(self.images.real_b)

    # def cycle_forward_image(self, real, gen_fwd):
    def cycle_forward_image(self, real, gen_fwd, gen_bkw):
        # pylint: disable=no-self-use
        # (N, C, H, W)
        fake = gen_fwd(real)
        reco = gen_bkw(fake)

        consist_fake = None

        if self.consist_model is not None:
            consist_fake = self.consist_model(fake)

        # return (fake, consist_fake)
        return (fake, reco, consist_fake)

    def idt_forward_image(self, real, gen):
        # pylint: disable=no-self-use
        # (N, C, H, W)
        idt = gen(real)
        return idt

    def forward_dispatch(self, direction):
        if direction == 'ab':
            (
                # self.images.fake_b,
                self.images.fake_b, self.images.reco_a,
                self.images.consist_fake_b
            ) = self.cycle_forward_image(
                # self.images.real_a, self.models.gen_ab
                self.images.real_a, self.models.gen_ab, self.models.gen_ba
            )

        elif direction == 'ba':
            (
                self.images.fake_a, self.images.reco_b,
                self.images.consist_fake_a
            ) = self.cycle_forward_image(
                self.images.real_b, self.models.gen_ba, self.models.gen_ab
            )

        elif direction == 'aa':
            self.images.idt_a = \
                self.idt_forward_image(self.images.real_a, self.models.gen_ba)

        elif direction == 'bb':
            self.images.idt_b = \
                self.idt_forward_image(self.images.real_b, self.models.gen_ab)

        elif direction == 'avg-ab':
            (
                # self.images.fake_b,
                self.images.fake_b, self.images.reco_a,
                self.images.consist_fake_b
            ) = self.cycle_forward_image(
                self.images.real_a,
                # self.models.avg_gen_ab
                self.models.avg_gen_ab, self.models.avg_gen_ba
            )

        elif direction == 'avg-ba':
            (
                self.images.fake_a, self.images.reco_b,
                self.images.consist_fake_a
            ) = self.cycle_forward_image(
                self.images.real_b,
                self.models.avg_gen_ba, self.models.avg_gen_ab
            )

        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")

    def forward(self):
        if self.images.real_a is not None:
            if self.avg_momentum is not None:
                self.forward_dispatch(direction = 'avg-ab')
            else:
                self.forward_dispatch(direction = 'ab')

        if self.images.real_b is not None:
            if self.avg_momentum is not None:
                self.forward_dispatch(direction = 'avg-ba')
            else:
                self.forward_dispatch(direction = 'ba')

    def eval_consist_loss(
        # self, consist_real_0, consist_fake_1
        self, consist_real_0, consist_fake_1, lambda_cycle_0
    ):
        # return self.lambda_consist * self.criterion_consist(
        return lambda_cycle_0 * self.lambda_consist * self.criterion_consist(
            consist_fake_1, consist_real_0
        )

    def eval_loss_of_cycle_a_forward(
        # self, disc_1, real_0, fake_1, fake_queue_1
        self, disc_b, real_a, fake_b, reco_a, fake_queue_b, lambda_cycle_a
    ):
        # pylint: disable=too-many-arguments
        # NOTE: Queue is updated in discriminator backprop
        disc_pred_fake_b = queued_forward(
            disc_b, fake_b, fake_queue_b, update_queue = False
        )

        loss_gen   = self.criterion_gan(disc_pred_fake_b, True)
        loss_cycle_a = lambda_cycle_a * self.criterion_cycle_a(reco_a, real_a)

        # loss = loss_gen
        loss = loss_gen + loss_cycle_a

        # return (loss_gen, loss)
        return (loss_gen, loss_cycle_a, loss)
    
    def eval_loss_of_cycle_b_forward(
        self, disc_a, real_b, fake_a, reco_b, fake_queue_a, lambda_cycle_b
    ):
        disc_pred_fake_a = queued_forward(
            disc_a, fake_a, fake_queue_a, update_queue = False
        )

        loss_gen   = self.criterion_gan(disc_pred_fake_a, True)
        loss_cycle_b = lambda_cycle_b * self.criterion_cycle_b(reco_b, real_b)

        loss = loss_gen + loss_cycle_b

        return (loss_gen, loss_cycle_b, loss)

    # def eval_loss_of_idt_forward(self, real_0, idt_0):
    def eval_loss_of_idt_forward(self, real_0, idt_0, lambda_cycle_0):
        loss_idt = (
              lambda_cycle_0
            * self.lambda_idt
            * self.criterion_idt(idt_0, real_0)
        )

        loss = loss_idt

        return (loss_idt, loss)

    def eval_iou_loss(self, fake, real):
        iou_loss = self.iou_loss(fake, real)
        return self.lambda_iou * iou_loss

    def eval_dice_loss(self, fake, real):
        dice_loss = self.dice_loss(fake, real)
        return self.lambda_dice * dice_loss


    def backward_gen(self, direction):
        if direction == 'ab':
            # (self.losses.gen_ab, loss) \
            (self.losses.gen_ab, self.losses.cycle_a, loss) \
                = self.eval_loss_of_cycle_a_forward(
                    self.models.disc_b,
                    # self.images.real_a, self.images.fake_b,
                    self.images.real_a, self.images.fake_b, self.images.reco_a,
                    # self.queues.fake_b
                    self.queues.fake_b, self.lambda_a
                )

            if self.consist_model is not None:
                self.losses.consist_a = self.eval_consist_loss(
                    self.images.consist_real_a, self.images.consist_fake_b,
                    self.lambda_a
                )

                loss += self.losses.consist_a
            
            self.losses.iou = self.eval_iou_loss(self.images.fake_b, self.images.real_b)
            loss += self.losses.iou

            self.losses.dice = self.eval_dice_loss(self.images.fake_b, self.images.real_b)
            loss += self.losses.dice

        elif direction == 'ba':
            (self.losses.gen_ba, self.losses.cycle_b, loss) \
                = self.eval_loss_of_cycle_b_forward(
                    self.models.disc_a,
                    self.images.real_b, self.images.fake_a, self.images.reco_b,
                    self.queues.fake_a, self.lambda_b
                )

            if self.consist_model is not None:
                self.losses.consist_b = self.eval_consist_loss(
                    self.images.consist_real_b, self.images.consist_fake_a,
                    self.lambda_b
                )

                loss += self.losses.consist_b

        elif direction == 'aa':
            (self.losses.idt_a, loss) \
                = self.eval_loss_of_idt_forward(
                    self.images.real_a, self.images.idt_a, self.lambda_a
                )

        elif direction == 'bb':
            (self.losses.idt_b, loss) \
                = self.eval_loss_of_idt_forward(
                    # self.images.real_b, self.images.idt_b
                    self.images.real_b, self.images.idt_b, self.lambda_b
                )

        else:
            raise ValueError(f"Unknown forward direction: '{direction}'")


        loss.backward()

    def backward_discriminator_base(
        self, model, real, fake, queue_real, queue_fake
    ):
        # pylint: disable=too-many-arguments
        loss_gp = None

        if self.gp is not None:
            loss_gp = self.gp(
                model, fake, real,
                model_kwargs_fake = { 'extra_bodies' : queue_fake.query() },
                model_kwargs_real = { 'extra_bodies' : queue_real.query() },
            )
            loss_gp.backward()

        pred_real = queued_forward(
            model, real, queue_real, update_queue = True
        )
        loss_real = self.criterion_gan(pred_real, True)

        pred_fake = queued_forward(
            model, fake, queue_fake, update_queue = True
        )
        loss_fake = self.criterion_gan(pred_fake, False)

        loss = (loss_real + loss_fake) * 0.5
        loss.backward()

        return (loss_gp, loss)

    def backward_discriminators(self):
        fake_a = self.images.fake_a.detach()
        fake_b = self.images.fake_b.detach()

        loss_gp_b, self.losses.disc_b \
            = self.backward_discriminator_base(
                self.models.disc_b, self.images.real_b, fake_b,
                self.queues.real_b, self.queues.fake_b
            )

        if loss_gp_b is not None:
            self.losses.gp_b = loss_gp_b

        loss_gp_a, self.losses.disc_a = \
            self.backward_discriminator_base(
                self.models.disc_a, self.images.real_a, fake_a,
                self.queues.real_a, self.queues.fake_a
            )

        if loss_gp_a is not None:
            self.losses.gp_a = loss_gp_a

    def optimization_step_gen(self):
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], False)
        # self.set_requires_grad([self.models.disc_b], False)
        self.optimizers.gen.zero_grad(set_to_none = True)

        # dir_list = [ 'ab']
        dir_list = [ 'ab', 'ba' ]
        if self.lambda_idt > 0:
            # dir_list += [ 'bb' ]
            dir_list += [ 'aa', 'bb' ]

        for direction in dir_list:
            self.forward_dispatch(direction)
            self.backward_gen(direction)

        self.optimizers.gen.step()

    def optimization_step_disc(self):
        self.set_requires_grad([self.models.disc_a, self.models.disc_b], True)
        # self.set_requires_grad([self.models.disc_b], True)
        self.optimizers.disc.zero_grad(set_to_none = True)

        self.backward_discriminators()

        self.optimizers.disc.step()

    def _accumulate_averages(self):
        update_average_model(
            self.models.avg_gen_ab, self.models.gen_ab, self.avg_momentum
        )
        update_average_model(
            self.models.avg_gen_ba, self.models.gen_ba, self.avg_momentum
        )

    def optimization_step(self):
        self.optimization_step_gen()
        self.optimization_step_disc()

        if self.avg_momentum is not None:
            self._accumulate_averages()
