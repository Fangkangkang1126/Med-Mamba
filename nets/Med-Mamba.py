# from nnunetv2.nets.network_architecture.custom_modules.custom_networks.CoTr.ResTransUNet import ResTranUnet
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from nnunetv2.nets.basicblocks import *
from nnunetv2.nets.blocks import MambaLayer,CBAM, MutiFuse

from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from dynamic_network_architectures.building_blocks.helper import  convert_dim_to_conv_op
from nnunetv2.nets.network_architecture.initialization import InitWeights_He

from nnunetv2.nets.load_weight_med import upkern_load_weights







class Med_Mamba(nn.Module):
    """
        planing use to sconstruct network of ssm and little attention ...  
        encoder : BasicBlock MambaLayer
        skip : add a attention of channel and spatial block 
        bottle: Basic Block and attention of channel and spatial block 
        
    
    """

    def __init__(self,
                    in_channels: int,
                    n_channels: int,
                    n_classes: int,
                    exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                    kernel_size: int = 7,  # Ofcourse can test kernel_size
                    enc_kernel_size: int = None,
                    dec_kernel_size: int = None,
                    deep_supervision: bool = False,  # Can be used to test deep supervision
                    do_res: bool = False,  # Can be used to individually test residual connection
                    do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                    checkpoint_style: bool = None,  # Either inside block or outside block
                    block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test staging ratio:
                    # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                    norm_type='group',
                    dim='3d',  # 2d or 3d
                    grn=False
                    ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d

        self.stem = conv(in_channels, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            BasicBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[0])])
        
        self.atten0 = CBAM(gate_channels=n_channels)

        self.down_0 = BasicDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )
        
        ###stage1
        self.mamba1 = MambaLayer(2 * n_channels)
        self.enc_block_1 = nn.Sequential(*[
            BasicBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[1])]
                                         )
        # self.atten1 = CBAM(gate_channels=n_channels * 2)

        self.down_1 = BasicDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        
        ### stage2
        self.mamba2 = MambaLayer(4 * n_channels)
        self.enc_block_2 = nn.Sequential(*[
            BasicBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[2])])
        # self.atten2 = CBAM(gate_channels=n_channels * 4)

        self.down_2 = BasicDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        
        ###stage3
        self.mamba3 = MambaLayer(8 * n_channels)

        self.enc_block_3 = nn.Sequential(*[
            BasicBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[3])]
                                            )
        # self.atten3 = CBAM(gate_channels=n_channels *  8)

        self.down_3 = BasicDownBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        
        ##stage4
        self.mamba4 = MambaLayer(16 * n_channels)
        self.Convbottleneck = nn.Sequential(*[
            BasicBlock(
                in_channels=n_channels * 16,
                out_channels=n_channels * 16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[4])]
                                        )
        self.attenBottle = CBAM(gate_channels=n_channels * 16)##  BottleBlock

        self.up_3 = BasicUpBlock(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        
        self.msf3 = MutiFuse(n_channels *8,n_channels * 8,dim='3d',n_groups=n_channels * 8)
        self.dec_block_3 = nn.Sequential(*[
            BasicBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[5])]
                                         )

        self.up_2 = BasicUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.msf2 = MutiFuse(n_channels * 4,n_channels * 4,dim='3d',n_groups=n_channels * 4)
        self.dec_block_2 = nn.Sequential(*[
            BasicBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[6])]
                                         )

        self.up_1 = BasicUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.msf1 = MutiFuse(n_channels * 2,n_channels * 2,dim='3d',n_groups=n_channels * 2)
        self.dec_block_1 = nn.Sequential(*[
            BasicBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[7])]
                                         )

        self.up_0 = BasicUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.msf0 = MutiFuse(n_channels * 1,n_channels * 1,dim='3d',n_groups=n_channels * 1)
        self.dec_block_0 = nn.Sequential(*[
            BasicBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[8])])

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=n_channels * 16, n_classes=n_classes, dim=dim)

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor,use_reentrant=False)
        return x

    def forward(self, x):

        x = self.stem(x)
        
        x_res_0 = self.enc_block_0(x)
        # x_skip_0 = self.atten0(x_res_0)  #skip           
        x = self.down_0(x_res_0)
        
        #stage1
        x_res_1 = self.mamba1(x)
        x_res_1 = self.enc_block_1(x_res_1)
        # x_skip_1 = self.atten1(x_res_1)  #skip
        x = self.down_1(x_res_1)
        
        #stage2
        x_res_2 = self.mamba2(x)
        x_res_2 = self.enc_block_2(x_res_2)
        # x_skip_2 = self.atten2(x_res_2)  #skip
        x = self.down_2(x_res_2)
        
        #stage3
        x_res_3 = self.mamba3(x)
        x_res_3 = self.enc_block_3(x_res_3)
        # x_skip_3 = self.atten3(x_res_3)  #skip
        x = self.down_3(x_res_3)

        x_bot = self.Convbottleneck(x)
        x = self.attenBottle(x_bot)
        
        if self.do_ds:
            x_ds_4 = self.out_4(x)

        x_up_3 = self.up_3(x)
        dec_x = self.msf3(x_up_3,x_res_3) #x_skip_3 + x_up_3
        x = self.dec_block_3(dec_x)

        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        dec_x = self.msf2(x_up_2,x_res_2) # x_skip_2 + x_up_2
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = self.msf1(x_up_1,x_res_1) #x_skip_1 + x_up_1
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = self.msf0(x_up_0,x_res_0) # x_skip_0 + x_up_0
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x


def get_Med_Mamba_k5_from_plans(plans_manager: PlansManager,
                           dataset_json: dict,
                           configuration_manager: ConfigurationManager,
                           num_input_channels: int,
                           deep_supervision: bool = True):
    """
    we may have to change this in the future to accommodate other plans -> network mappings

    num_input_channels can differ depending on whether we do cascade. Its best to make this info available in the
    trainer rather than inferring it again from the plans here.
    """
    num_stages = len(configuration_manager.conv_kernel_sizes)

    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'Med_Mamba'
    network_class = Med_Mamba
    # kwargs = {
    #     'MedAtssmv5': {
    #         'conv_bias': True,
    #         'norm_op': get_matching_instancenorm(conv_op),
    #         'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
    #         'dropout_op': None, 'dropout_op_kwargs': None,
    #         'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
    #     }
    # }

    # conv_or_blocks_per_stage = {
    #     'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
    #     'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    # }

    model3 = network_class(
            in_channels=2,
            n_channels=32,
            n_classes=label_manager.num_segmentation_heads,
            exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
            # exp_r = 2,
            kernel_size=3,  # Can test kernel_size
            deep_supervision=True,  # Can be used to test deep supervision
            do_res=True,  # Can be used to individually test residual connection
            do_res_up_down=True,
            block_counts = [2,2,2,2,2,2,2,2,2],
            # block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
            checkpoint_style='outside_block',
            dim='3d',
            grn=True

        )
    # model3.apply(InitWeights_He(1e-2))
    model3.load_state_dict(torch.load("/mnt/data/fkk/DATASET/nnUNetFrame/DATASET/nnUNet_results/......."))
    model5 =  network_class(
                in_channels=2,
                n_channels=32,
                n_classes=label_manager.num_segmentation_heads,
                exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
                # exp_r = 2,
                kernel_size=5,  # Can test kernel_size
                deep_supervision=True,  # Can be used to test deep supervision
                do_res=True,  # Can be used to individually test residual connection
                do_res_up_down=True,
                block_counts = [2,2,2,2,2,2,2,2,2],
                # block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
                checkpoint_style='outside_block',
                dim='3d',
                grn=True

            )
    model = upkern_load_weights(model5,model3)
    return model

if __name__ == "__main__":
    network = Med_Mamba(
        in_channels=2,
        n_channels=32,
        n_classes=3,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
        # exp_r = 2,
        kernel_size=3,  # Can test kernel_size
        deep_supervision=True,  # Can be used to test deep supervision
        do_res=True,  # Can be used to individually test residual connection
        do_res_up_down=True,
        block_counts = [2,2,2,2,2,2,2,2,2],
        # block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        # checkpoint_style='outside_block',
        dim='3d',
        grn=True

    ).cuda()
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(count_parameters(network))

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    # x = torch.zeros((1, 2, 128, 128, 128), requires_grad=False).cuda()
    # flops = FlopCountAnalysis(network, x)
    # print(flops.total())

    with torch.no_grad():
        print(network)
        # print(model)
        x = torch.zeros((1, 2, 128, 128, 128)).cuda()
        # print(model(x).shape)
        output = network(x)
        for idx in range(len(output)):
            print(output[idx].shape)
