fn main() {
    let paths = vec!["./cpp/UFront2TOSA/build/lib", "./lib", "/lib/x86_64-linux-gnu/"];
    for path in paths {
        println!("cargo:rustc-link-search=native={}", path);
    }
    
    println!("cargo:rustc-link-lib=dylib=UfrontCAPI");
    println!("cargo:rerun-if-changed=build.rs");

}
