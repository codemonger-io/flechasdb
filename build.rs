fn main() {
    println!("cargo:rerun-if-changed=src/protos/database.proto");
    protobuf_codegen::Codegen::new()
        .protoc()
        .protoc_path(&protoc_bin_vendored::protoc_bin_path().unwrap())
        .includes(&["src/protos"])
        .input("src/protos/database.proto")
        .cargo_out_dir("protos")
        .run_from_script();
}
