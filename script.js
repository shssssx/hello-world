const levels = [
  {
    id: 'L1',
    name: 'Level 1 基础识别',
    intro: '目标：识别概念与简单推理。',
    questions: [
      {
        id: 'L1Q1',
        type: 'mcq',
        prompt: '“所有猫都是动物，花花是猫”可以推出什么？',
        options: ['花花是植物', '花花是动物', '无法判断'],
        answer: 1,
        hint: '这是最经典的三段论。',
        tag: '演绎'
      },
      {
        id: 'L1Q2',
        type: 'text',
        prompt: '把“如果下雨就带伞”改写为“_____ 时要带伞”。',
        answer: '下雨',
        hint: '找条件部分。',
        tag: '条件语句'
      },
      {
        id: 'L1Q3',
        type: 'mcq',
        prompt: '哪个更像“事实陈述”而非“观点”？',
        options: ['我觉得这电影很好看', '今天气温是26°C', '这家店最棒'],
        answer: 1,
        hint: '能被测量或查证。',
        tag: '信息判断'
      }
    ]
  },
  {
    id: 'L2',
    name: 'Level 2 结构化思考',
    intro: '目标：分解问题、提取关键信息。',
    questions: [
      {
        id: 'L2Q1',
        type: 'mcq',
        prompt: '解复杂题第一步通常应做什么？',
        options: ['直接套公式', '先明确已知与目标', '先猜答案'],
        answer: 1,
        hint: '先建模，再求解。',
        tag: '问题分解'
      },
      {
        id: 'L2Q2',
        type: 'text',
        prompt: '“把大任务切成小步骤”对应的方法叫“任务____”。',
        answer: '分解',
        hint: '两个字。',
        tag: '问题分解'
      },
      {
        id: 'L2Q3',
        type: 'order',
        prompt: '将解题流程按合理顺序排序（填1~4）：理解题意、验证结果、执行步骤、制定方案。',
        items: ['理解题意', '验证结果', '执行步骤', '制定方案'],
        answer: [1, 4, 3, 2],
        hint: '先理解，再计划，再执行，最后检查。',
        tag: '流程意识'
      }
    ]
  },
  {
    id: 'L3',
    name: 'Level 3 迁移与应用',
    intro: '目标：把思路迁移到新情境。',
    questions: [
      {
        id: 'L3Q1',
        type: 'mcq',
        prompt: '你在做预算，发现支出超标。最优先做什么？',
        options: ['继续消费', '识别高占比支出项', '借钱填补'],
        answer: 1,
        hint: '抓主要矛盾。',
        tag: '决策'
      },
      {
        id: 'L3Q2',
        type: 'text',
        prompt: '“先抓80%影响因素”常对应“二八____”原则。',
        answer: '法则',
        hint: '二八后面常接什么？',
        tag: '优先级'
      },
      {
        id: 'L3Q3',
        type: 'mcq',
        prompt: '当两个方案都可行时，最稳妥的比较方式是？',
        options: ['选更快但风险未知的', '列出成本、收益、风险进行对比', '随机决定'],
        answer: 1,
        hint: '多维评估。',
        tag: '方案评估'
      }
    ]
  },
  {
    id: 'L4',
    name: 'Level 4 综合挑战',
    intro: '目标：在不完整信息下做可解释决策。',
    questions: [
      {
        id: 'L4Q1',
        type: 'mcq',
        prompt: '项目延期且需求变化频繁，哪种策略更合理？',
        options: ['一次性重做全部', '小步迭代+每周复盘', '停止沟通'],
        answer: 1,
        hint: '动态环境中需要反馈闭环。',
        tag: '迭代'
      },
      {
        id: 'L4Q2',
        type: 'text',
        prompt: '“提出假设→做小实验→根据结果调整”属于“____循环”。',
        answer: '反馈',
        hint: '两个字。',
        tag: '实验思维'
      },
      {
        id: 'L4Q3',
        type: 'mcq',
        prompt: '哪句话体现了“可解释决策”？',
        options: ['我拍脑袋选的', '基于A/B测试与成本约束，我选择方案B', '大家都这么选'],
        answer: 1,
        hint: '能说明依据。',
        tag: '论证'
      }
    ]
  }
];

const STORAGE_KEY = 'exercise-workshop-answers-v1';
const state = {
  levelId: levels[0].id,
  answers: JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
};

const levelSelect = document.querySelector('#levelSelect');
const exerciseContainer = document.querySelector('#exerciseContainer');
const completionEl = document.querySelector('#completion');
const accuracyEl = document.querySelector('#accuracy');
const adviceEl = document.querySelector('#advice');
const template = document.querySelector('#questionTemplate');

document.querySelector('#checkAllBtn').addEventListener('click', () => {
  const current = levels.find((l) => l.id === state.levelId);
  current.questions.forEach((q) => checkQuestion(q.id));
  updateProgress();
});

document.querySelector('#resetBtn').addEventListener('click', () => {
  const current = levels.find((l) => l.id === state.levelId);
  current.questions.forEach((q) => delete state.answers[q.id]);
  persist();
  renderLevel(state.levelId);
});

function init() {
  levels.forEach((level) => {
    const option = document.createElement('option');
    option.value = level.id;
    option.textContent = `${level.name}｜${level.intro}`;
    levelSelect.append(option);
  });

  levelSelect.addEventListener('change', (event) => {
    state.levelId = event.target.value;
    renderLevel(state.levelId);
  });

  renderLevel(state.levelId);
}

function renderLevel(levelId) {
  const level = levels.find((l) => l.id === levelId);
  levelSelect.value = levelId;
  exerciseContainer.innerHTML = '';

  level.questions.forEach((q, index) => {
    const node = template.content.cloneNode(true);
    const card = node.querySelector('.card');
    card.dataset.qid = q.id;
    node.querySelector('.card__title').textContent = `第 ${index + 1} 题`;
    node.querySelector('.card__desc').textContent = q.prompt;
    const content = node.querySelector('.card__content');

    if (q.type === 'mcq') {
      q.options.forEach((opt, i) => {
        const label = document.createElement('label');
        const input = document.createElement('input');
        input.type = 'radio';
        input.name = q.id;
        input.value = String(i);
        if (String(state.answers[q.id]) === String(i)) input.checked = true;
        input.addEventListener('change', () => saveAnswer(q.id, i));
        label.append(input, ` ${opt}`);
        content.append(label);
      });
    }

    if (q.type === 'text') {
      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = '请输入答案';
      input.value = state.answers[q.id] ?? '';
      input.addEventListener('input', () => saveAnswer(q.id, input.value.trim()));
      content.append(input);
    }

    if (q.type === 'order') {
      const wrapper = document.createElement('div');
      wrapper.className = 'order-list';
      q.items.forEach((item, idx) => {
        const row = document.createElement('label');
        const num = document.createElement('input');
        num.type = 'number';
        num.min = '1';
        num.max = String(q.items.length);
        num.value = state.answers[q.id]?.[idx] ?? '';
        num.addEventListener('input', () => {
          const existing = state.answers[q.id] || [];
          existing[idx] = Number(num.value);
          saveAnswer(q.id, existing);
        });
        row.append(num, item);
        wrapper.append(row);
      });
      content.append(wrapper);
    }

    node.querySelector('.check-btn').addEventListener('click', () => {
      checkQuestion(q.id);
      updateProgress();
    });
    node.querySelector('.hint-btn').addEventListener('click', () => {
      const feedback = card.querySelector('.feedback');
      feedback.className = 'feedback';
      feedback.textContent = `提示：${q.hint}`;
    });

    exerciseContainer.append(node);
  });

  updateProgress();
}

function saveAnswer(questionId, value) {
  state.answers[questionId] = value;
  persist();
  updateProgress();
}

function checkQuestion(questionId) {
  const question = levels.flatMap((l) => l.questions).find((q) => q.id === questionId);
  const card = [...document.querySelectorAll('.card')].find((c) => c.dataset.qid === questionId);
  const feedback = card.querySelector('.feedback');
  const ok = isCorrect(question, state.answers[questionId]);
  feedback.textContent = ok ? '回答正确 ✅' : '还不对，再想想或点提示 💡';
  feedback.className = `feedback ${ok ? 'ok' : 'bad'}`;
  return ok;
}

function isCorrect(question, value) {
  if (value === undefined || value === null || value === '') return false;
  if (question.type === 'text') {
    return String(value).replace(/\s+/g, '') === String(question.answer).replace(/\s+/g, '');
  }
  if (question.type === 'order') {
    if (!Array.isArray(value) || value.length !== question.answer.length) return false;
    return question.answer.every((n, i) => Number(value[i]) === n);
  }
  return Number(value) === question.answer;
}

function updateProgress() {
  const level = levels.find((l) => l.id === state.levelId);
  const answers = level.questions.map((q) => state.answers[q.id]);
  const done = answers.filter((a) => a !== undefined && a !== null && a !== '').length;
  const right = level.questions.filter((q) => isCorrect(q, state.answers[q.id])).length;

  completionEl.textContent = `${Math.round((done / level.questions.length) * 100)}%`;
  accuracyEl.textContent = `${Math.round((right / level.questions.length) * 100)}%`;

  if (done < 2) {
    adviceEl.textContent = '先继续作答，系统会根据错题标签给你个性建议。';
  } else {
    const wrongTags = level.questions
      .filter((q) => !isCorrect(q, state.answers[q.id]) && state.answers[q.id] !== undefined)
      .map((q) => q.tag);
    adviceEl.textContent = wrongTags.length
      ? `建议优先复习：${[...new Set(wrongTags)].join('、')}。`
      : '表现很好！可以挑战下一难度。';
  }
}

function persist() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.answers));
}

init();
