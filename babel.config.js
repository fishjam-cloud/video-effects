module.exports = {
  presets: [["@babel/preset-typescript", { onlyRemoveTypeImports: true }]],
  plugins: [["unplugin-typegpu/babel", { forceTgpuAlias: "tgpu" }]],
};
