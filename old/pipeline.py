#!/usr/bin/env python3

import argparse
import parse_aleph
import ai_models_aleph


def main():
    print('-'*50)
    parser = argparse.ArgumentParser(description="ALEPH ML Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Command: database ---
    parser_db = subparsers.add_parser("database", help="Run ALEPH database generation")
    parser_db.add_argument('-p', '--path', help="Path to database", required=True)
    parser_db.add_argument('-n', '--name', help="Name of the run", required=True)

    # --- Command: train ---
    parser_train = subparsers.add_parser("train", help="Train the AI model")
    parser_train.add_argument('-p', '--path', help="Path to training data", required=True)
    parser_train.add_argument('-n', '--name', help="Name of the model", required=True)

    # --- Command: predict ---
    parser_pred = subparsers.add_parser("predict", help="Run predictions using the AI model")
    parser_pred.add_argument('-p', '--path', help="Path to prediction data", required=True)
    parser_pred.add_argument('-n', '--name', help="Name of the model", required=True)

    # --- Command: classifier ---
    parser_pred = subparsers.add_parser("classifier", help="Train a classifier with AiLEPH embeddings and predict FoldSeek labels")
    parser_pred.add_argument('-p', '--path', help="Path to prediction data", required=True)
    parser_pred.add_argument('-n', '--name', help="Name of the model", required=True)

    # --- Command: superpose ---
    parser_sp = subparsers.add_parser("superpose", help="Superpose two PDB structures")
    parser_sp.add_argument('--reference', help="Reference PDB file", required=True)
    parser_sp.add_argument('--target', help="Target or moving PDB file", required=True)

    # --- Command: create graphs ---
    parser_cg = subparsers.add_parser("create_graphs", help="Create the graphs of the given PDB structures")
    parser_cg.add_argument('-s', '--selection', help="Path to the selection", required=True)
    parser_cg.add_argument('-p', '--path', help="Path to the database", required=True)
    parser_cg.add_argument('-d', '--destination', help="Path where graphs will be stored", required=True)

    # --- Command: compute superposition ---
    parser_csp = subparsers.add_parser("pairs_superposition", help="Superpose two PDB structures (tmscore)")
    parser_csp.add_argument('-d1', '--database1', help="Path to the database", required=True)
    parser_csp.add_argument('-d2', '--database2', help="Path to the database", required=True)
    parser_csp.add_argument('-s', '--selection', help="Path to the selection", required=True)

    # --- Command: create annotations ---
    parser_ca = subparsers.add_parser("create_annotations", help="Create annotations for the given PDB structures")
    parser_ca.add_argument('-d', '--database', help="Path to the database", required=True)
    parser_ca.add_argument('-n', '--name', help="Path of the resulting database", required=True)
    parser_ca.add_argument('-m', '--max-size', type=int, default=None,
                           help="Maximum PDB file size in bytes (omit to include all sizes)")

    args = parser.parse_args()

    if args.command == "database":
        parse_aleph.run_database_aleph(args.path, args.name)
    elif args.command == "train":
        ai_models_aleph.load_database(args.path, args.name, True, False)
    elif args.command == "predict":
        ai_models_aleph.load_database(args.path, args.name, False, True)
    elif args.command == "classifier":
        ai_models_aleph.load_database(args.path, args.name, False, False)
    elif args.command == "superpose":
        parse_aleph.superpose_aleph(args.reference, args.target)
    elif args.command == "create_graphs":
        parse_aleph.create_graphs_pdb(args.selection, args.path, args.destination)
    elif args.command == "pairs_superposition":
        parse_aleph.foldseek_superposition(args.database1, args.database2, args.selection, canon=True)
    elif args.command == "create_annotations":
        parse_aleph.create_annotations(args.database, args.name, args.max_size)


if __name__ == "__main__":
    main()
