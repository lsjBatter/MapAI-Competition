import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #############################################################
    # Mandatory arguments. DO NOT EDIT
    #############################################################
    parser.add_argument("--submission-path", required=True)
    parser.add_argument("--data-type", required=True, default="validation", help="validation or test")
    parser.add_argument("--task", type=int, default=3, help="Which task you are submitting for")

    #############################################################
    # CUSTOM ARGUMENTS GOES HERE
    #############################################################
    parser.add_argument('--path_size', default=384, type=int, metavar='SIZE',
                            help='patch size for image cropped from orig image (default: 384)')
    parser.add_argument('--BATCH_SIZE', default=14, type=int, metavar='SIZE',
                            help='batch size for prediction process (default: 14)')
    parser.add_argument('--area_perc', default=0.5, type=float, metavar='SIZE',
                            help='Area percentage (default: 0.5)')
    parser.add_argument('--input', default=3, type=int, metavar='channels',
                            help='input channels (default: 3)')      
    parser.add_argument('--output', default=2, type=int, metavar='channels',
                            help='output channels (default: 2)')  
    parser.add_argument('--lidar', default=None, type=bool, 
                            help='lidar (default: None)') 
    parser.add_argument('--img', default='/boot/data1/Li_data/data/competition/MapAI-Competition/validation/images', type=str,metavar='DIR',
                            help='Test image path')
    parser.add_argument('--gt', default='/boot/data1/Li_data/data/competition/MapAI-Competition/validation/label', type=str,metavar='DIR',
                            help='groundtruth  path')
    parser.add_argument('--backbone', default='efficientnet-b3', type=str,metavar='backbone',
                            help='backbone')
    parser.add_argument('--modelp_path', default='https://drive.google.com/file/d/1guwUhWMKbqeztGRC1y41wSDDB6Q7aVrB/view?usp=sharing', type=str,metavar='DIR',
                            help='model_p path')
    parser.add_argument('--save_path', default='./submission', type=str,metavar='DIR',
                            help='save path')
    args = parser.parse_args()

    #############################################################
    # CODE GOES HERE
    # Save results into: args.submission_path
    #############################################################
    from big_pre import main as evaluate_model
    if args.task == 1:

        evaluate_model(args=args)
        
    elif args.task == 2:
        args.modelp_path = 'https://drive.google.com/file/d/1x9PlLXrn7vFN6zQone8VXjt-Vze4aFeG/view?usp=sharing'
        args.backbone = 'efficientnet-b4'
        args.path_size = 224
        args.input = 1
        args.lidar = True
        args.modelp_path = './model_p/Unet-efficientnet-b4_224.pt'
        evaluate_model(args=args)

    exit(0)
