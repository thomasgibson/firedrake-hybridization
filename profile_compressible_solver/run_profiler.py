from firedrake.petsc import PETSc
from argparse import ArgumentParser
from driver import run_profliler
import sys


PETSc.Log.begin()

parser = ArgumentParser(description=("""
Profile of 3D compressible solver for the Euler equations (dry atmosphere).
"""), add_help=False)

parser.add_argument("--hybridization",
                    action="store_true",
                    help="Use a hybridized compressible solver.")

parser.add_argument("--model_degree",
                    default=0,
                    type=int,
                    action="store",
                    help="Model degree")

parser.add_argument("--model_family",
                    default="RTCF",
                    choices=["RTCF", "RT", "BDFM"],
                    help="Family of finite element spaces")

parser.add_argument("--mesh_degree",
                    default=3,
                    type=int,
                    action="store",
                    help="Coordinate space degree")

parser.add_argument("--cfl",
                    default=1.,
                    type=float,
                    action="store",
                    help="CFL number to run at (determines dt).")

parser.add_argument("--run_cfl_range",
                    action="store_true",
                    help="Run several solves over a range of CFLs.")

parser.add_argument("--refinements",
                    default=4,
                    type=int,
                    action="store",
                    help="Resolution scaling parameter.")

parser.add_argument("--flexsolver",
                    action="store_true",
                    help="Switch to flex-GMRES and AMG.")

parser.add_argument("--stronger_smoother",
                    action="store_true",
                    help="Change smoother iterations from 3 to 5.")

parser.add_argument("--layers",
                    default=64,
                    type=int,
                    action="store",
                    help="Number of vertical layers.")

parser.add_argument("--debug",
                    action="store_true",
                    help="Turn on KSP monitors")

parser.add_argument("--rtol",
                    default=1.0e-6,
                    type=float,
                    help="Rtolerance for the linear solver.")

parser.add_argument("--suppress_data_output",
                    action="store_true",
                    help="Suppress data output.")

parser.add_argument("--help",
                    action="store_true",
                    help="Show help.")

args, _ = parser.parse_known_args()


if args.help:
    help = parser.format_help()
    PETSc.Sys.Print("%s\n" % help)
    sys.exit(1)


if args.run_cfl_range:

    for cfl in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        run_profliler(hybridization=args.hybridization,
                      model_degree=args.model_degree,
                      model_family=args.model_family,
                      mesh_degree=args.mesh_degree,
                      cfl=cfl,
                      refinements=args.refinements,
                      layers=args.layers,
                      debug=args.debug,
                      rtol=args.rtol,
                      flexsolver=args.flexsolver,
                      stronger_smoother=args.stronger_smoother,
                      suppress_data_output=args.suppress_data_output)

else:
    run_profliler(hybridization=args.hybridization,
                  model_degree=args.model_degree,
                  model_family=args.model_family,
                  mesh_degree=args.mesh_degree,
                  cfl=args.cfl,
                  refinements=args.refinements,
                  layers=args.layers,
                  debug=args.debug,
                  rtol=args.rtol,
                  flexsolver=args.flexsolver,
                  stronger_smoother=args.stronger_smoother,
                  suppress_data_output=args.suppress_data_output)
