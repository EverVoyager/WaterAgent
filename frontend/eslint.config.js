import js from '@eslint/js'
import vue from 'eslint-plugin-vue'
import tseslint from 'typescript-eslint'
import globals from 'globals'

export default [
  // 忽略目录与编译产物
  {
    ignores: [
      'dist/',
      'node_modules/',
      'coverage/',
      'vite.config.js',
      'vite.config.d.ts',
    ],
  },

  // 基础推荐配置（tseslint 须在 vue 之前，避免其全局 parser 覆盖 Vue 解析器）
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...vue.configs['flat/recommended'],

  // 项目源码配置
  {
    files: ['src/**/*.vue', 'src/**/*.ts'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: {
        ...globals.browser,
        ...globals.node,
      },
      // Vue SFC 的 <script lang="ts"> 块使用 TS 解析器
      parserOptions: {
        parser: tseslint.parser,
      },
    },
    rules: {
      // 关闭基础 no-unused-vars，交由 @typescript-eslint 版本处理
      'no-unused-vars': 'off',
      // 忽略以下划线开头的参数 / 变量 / catch 参数
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      // 允许 console.warn / console.error，其余 console 调用告警
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      // API 响应类型广泛使用 any 表达灵活结构，降级为告警避免侵入式改动源码
      '@typescript-eslint/no-explicit-any': 'warn',
      // Vue SFC 类型 shim 中的 {} 空对象类型降级为告警
      '@typescript-eslint/no-empty-object-type': 'warn',
      // Vue 组件名在模板中按 PascalCase
      'vue/component-name-in-template-casing': ['error', 'PascalCase'],
      // App.vue 为单字根组件，豁免多词组件名约束
      'vue/multi-word-component-names': 'off',
      // 消息对象（reactive 引用）持有 UI 状态（chainExpanded 等），v-model 直接修改为既有设计，降级告警避免侵入式重构
      'vue/no-mutating-props': 'warn',
    },
  },
]
